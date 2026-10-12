// SPDX-License-Identifier: AGPL-3.0-only
//! Optional loopback inference. Network I/O occurs only during host polling.
use nextime_ranker::Candidate;
use serde_json::{Value, json};
use std::{
    io::{ErrorKind, Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct LocalLlm {
    token: String,
    request: Option<Value>,
    job: Option<String>,
    started: Instant,
    transport: Option<Transport>,
    indices: Vec<usize>,
}
#[derive(Debug)]
struct Transport {
    socket: TcpStream,
    packet: Vec<u8>,
    sent: usize,
    response: Vec<u8>,
    started: Instant,
    submit: bool,
}
/// Non-sensitive configuration diagnostics suitable for the candidate UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalConfigError {
    MissingDirectory,
    MissingFile,
    ReadFailed,
    InvalidJson,
    Disabled,
    InvalidToken,
}
impl LocalConfigError {
    pub fn status(self) -> &'static str {
        match self {
            Self::MissingDirectory => "local:directory-missing",
            Self::MissingFile => "local:config-missing",
            Self::ReadFailed => "local:config-unreadable",
            Self::InvalidJson => "local:config-invalid",
            Self::Disabled => "local:config-disabled",
            Self::InvalidToken => "local:token-invalid",
        }
    }
}
#[cfg(windows)]
fn configuration_diagnostic(directory: &std::path::Path, stage: &str, code: Option<i32>) {
    // Local, bounded per-process diagnostic. Never include configuration contents,
    // authentication tokens, candidates or preceding text.
    let diagnostic = json!({"directory":directory.to_string_lossy(), "stage":stage,
        "os_error":code, "pid":std::process::id()});
    let path = std::env::temp_dir().join(format!("luqnexa-ai-config-{}.json", std::process::id()));
    if let Ok(bytes) = serde_json::to_vec(&diagnostic) {
        let _ = std::fs::write(path, bytes);
    }
}
#[cfg(windows)]
fn registry_configuration(directory: &std::path::Path) -> Option<Vec<u8>> {
    use std::ffi::c_void;
    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(
            key: *mut c_void,
            subkey: *const u16,
            value: *const u16,
            flags: u32,
            kind: *mut u32,
            data: *mut c_void,
            size: *mut u32,
        ) -> i32;
    }
    let key = (-2147483647isize) as *mut c_void; // HKEY_CURRENT_USER
    let subkey: Vec<u16> = "Software\\Luqnexa\\LocalAI\0".encode_utf16().collect();
    let value: Vec<u16> = "Configuration\0".encode_utf16().collect();
    let mut size = 0;
    // REG_SZ only, with a fixed upper bound before allocation. No key creation.
    let result = unsafe {
        RegGetValueW(
            key,
            subkey.as_ptr(),
            value.as_ptr(),
            2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut size,
        )
    };
    if result != 0 || !(2..=8192).contains(&size) || size % 2 != 0 {
        configuration_diagnostic(directory, "registry-size", Some(result));
        return None;
    }
    let mut data = vec![0u16; size as usize / 2];
    let result = unsafe {
        RegGetValueW(
            key,
            subkey.as_ptr(),
            value.as_ptr(),
            2,
            std::ptr::null_mut(),
            data.as_mut_ptr().cast(),
            &mut size,
        )
    };
    if result != 0 || size as usize > data.len() * 2 || size % 2 != 0 {
        configuration_diagnostic(directory, "registry-read", Some(result));
        return None;
    }
    data.truncate(size as usize / 2);
    let bytes = String::from_utf16(&data)
        .ok()?
        .trim_end_matches('\0')
        .as_bytes()
        .to_vec();
    let config = registry_bytes_for_directory(bytes, directory);
    if config.is_none() {
        configuration_diagnostic(directory, "registry-directory-or-json", None);
    }
    config
}
#[cfg(any(windows, test))]
fn registry_bytes_for_directory(bytes: Vec<u8>, directory: &std::path::Path) -> Option<Vec<u8>> {
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    // Do not apply another profile's mirror or override an isolated config directory.
    if std::path::Path::new(value["directory"].as_str()?) != directory {
        return None;
    }
    Some(bytes)
}
impl Transport {
    fn start(token: &str, path: &str, body: &Value, submit: bool) -> Option<Self> {
        // Fixed loopback only. The connection handshake is capped at 10 ms;
        // all subsequent transfer is nonblocking and runs on host timer ticks.
        let address: SocketAddr = "127.0.0.1:43821".parse().ok()?;
        let socket = TcpStream::connect_timeout(&address, Duration::from_millis(10)).ok()?;
        socket.set_nodelay(true).ok()?;
        socket.set_nonblocking(true).ok()?;
        let body = serde_json::to_vec(body).ok()?;
        let header = format!(
            "POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut packet = header.into_bytes();
        packet.extend_from_slice(&body);
        Some(Self {
            socket,
            packet,
            sent: 0,
            response: Vec::new(),
            started: Instant::now(),
            submit,
        })
    }
    fn step(&mut self) -> Option<Option<Value>> {
        if self.started.elapsed() > Duration::from_secs(2) {
            return Some(None);
        }
        while self.sent < self.packet.len() {
            match self.socket.write(&self.packet[self.sent..]) {
                Ok(0) => return Some(None),
                Ok(n) => self.sent += n,
                Err(e) if e.kind() == ErrorKind::WouldBlock => return None,
                Err(_) => return Some(None),
            }
        }
        let mut buffer = [0u8; 1024];
        loop {
            match self.socket.read(&mut buffer) {
                Ok(0) => {
                    if !self.response.starts_with(b"HTTP/1.0 200 ") {
                        return Some(None);
                    }
                    let Some(end) = self.response.windows(4).position(|s| s == b"\r\n\r\n") else {
                        return Some(None);
                    };
                    return Some(serde_json::from_slice(&self.response[end + 4..]).ok());
                }
                Ok(n) => {
                    self.response.extend_from_slice(&buffer[..n]);
                    if self.response.len() > 8192 {
                        return Some(None);
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => return None,
                Err(_) => return Some(None),
            }
        }
    }
}
impl LocalLlm {
    pub fn pending(&self) -> bool {
        self.request.is_some() || self.job.is_some() || self.transport.is_some()
    }
    pub fn configured() -> Option<Self> {
        Self::try_configured().ok()
    }
    pub fn try_configured() -> Result<Self, LocalConfigError> {
        let directory =
            std::env::var_os("LOCALAPPDATA").ok_or(LocalConfigError::MissingDirectory)?;
        let directory = std::path::PathBuf::from(directory);
        let path = directory.join("Luqnexa/local-ai.json");
        let bytes = std::fs::read(path).map_err(|error| {
            #[cfg(windows)]
            configuration_diagnostic(&directory, "file-read", error.raw_os_error());
            if error.kind() == ErrorKind::NotFound {
                LocalConfigError::MissingFile
            } else {
                LocalConfigError::ReadFailed
            }
        });
        #[cfg(windows)]
        let bytes = match bytes {
            Err(LocalConfigError::MissingFile) => {
                registry_configuration(&directory).ok_or(LocalConfigError::MissingFile)
            }
            other => other,
        };
        let bytes = bytes?;
        let config: Value =
            serde_json::from_slice(&bytes).map_err(|_| LocalConfigError::InvalidJson)?;
        if config["enabled"] != true {
            return Err(LocalConfigError::Disabled);
        }
        let token = config["token"]
            .as_str()
            .ok_or(LocalConfigError::InvalidToken)?;
        if token.len() != 64 || !token.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(LocalConfigError::InvalidToken);
        }
        Ok(Self {
            token: token.into(),
            request: None,
            job: None,
            started: Instant::now(),
            transport: None,
            indices: Vec::new(),
        })
    }
    pub fn cancel(&mut self) {
        self.request = None;
        self.job = None;
        self.transport = None;
        self.indices.clear();
    }
    pub fn begin(&mut self, context: &str, reading: &str, candidates: &[Candidate]) {
        self.cancel();
        let literal_tokens = Self::ascii_tokens(reading);
        let exclude_literal = literal_tokens.is_empty()
            || (!reading.is_ascii()
                && candidates
                    .iter()
                    .any(|c| !c.surface.eq_ignore_ascii_case(reading)));
        if (context.is_empty() && literal_tokens.is_empty())
            || context.chars().count() > 96
            || reading.is_empty()
            || reading.chars().count() > 128
            || candidates.len() < 2
        {
            return;
        }
        // Keep the literal fallback available to the user, but do not ask a
        // model to prefer it over actual dictionary spellings.
        self.indices = candidates
            .iter()
            .take(12)
            .enumerate()
            .filter(|(_, c)| {
                (!exclude_literal || !c.surface.eq_ignore_ascii_case(reading))
                    && !c.surface.is_empty()
                    && c.surface.chars().count() <= 256
                    && (literal_tokens.is_empty() || {
                        let tokens = Self::ascii_tokens(&c.surface);
                        tokens.len() == literal_tokens.len()
                            && tokens
                                .iter()
                                .zip(&literal_tokens)
                                .all(|(a, b)| a.eq_ignore_ascii_case(b))
                    })
            })
            .map(|(i, _)| i)
            .collect();
        if self.indices.len() < 2 {
            self.indices.clear();
            return;
        }
        let context = if context.is_empty() {
            "No preceding context. Choose conventional spelling."
        } else {
            context
        };
        self.request = Some(json!({"context":context, "reading":reading,
            "candidates":self.indices.iter().map(|&i| &candidates[i].surface).collect::<Vec<_>>() }));
        self.started = Instant::now();
    }
    // English names and numbers already present in the normalized reading
    // are literal input. A model may rank Japanese spellings around them,
    // but cannot delete, duplicate or substitute them. Casing is model-selected.
    fn ascii_tokens(text: &str) -> Vec<&str> {
        text.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .collect()
    }
    pub fn poll(&mut self, candidates: &mut Vec<Candidate>) -> bool {
        if self.started.elapsed() > Duration::from_secs(30) {
            self.cancel();
            return false;
        }
        if self.transport.is_none() {
            self.transport = if let Some(request) = self.request.take() {
                Transport::start(&self.token, "/submit", &request, true)
            } else if let Some(job) = &self.job {
                Transport::start(&self.token, "/poll", &json!({"job":job}), false)
            } else {
                return false;
            };
            if self.transport.is_none() {
                self.cancel();
                return false;
            }
        }
        let transport = self.transport.as_mut().unwrap();
        let submit = transport.submit;
        let Some(reply) = transport.step() else {
            return false;
        };
        self.transport = None;
        let Some(reply) = reply else {
            self.cancel();
            return false;
        };
        if submit {
            self.job = reply["job"]
                .as_str()
                .filter(|s| s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                .map(str::to_owned);
            return false;
        }
        if reply["ready"] != true {
            return false;
        }
        self.job = None;
        let Some(index) = reply["index"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
        else {
            return false;
        };
        let Some(&index) = self.indices.get(index) else {
            return false;
        };
        Self::select(index, candidates)
    }
    fn select(index: usize, candidates: &mut Vec<Candidate>) -> bool {
        if index == 0 || index >= candidates.len().min(12) {
            return false;
        }
        let mut selected = candidates.remove(index);
        selected.score = candidates
            .iter()
            .map(|c| c.score)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        candidates.insert(0, selected);
        true
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn registry_mirror_requires_the_same_configuration_directory() {
        use super::registry_bytes_for_directory;
        let bytes = br#"{"enabled":true,"directory":"synthetic-profile"}"#.to_vec();
        assert!(
            registry_bytes_for_directory(bytes.clone(), std::path::Path::new("synthetic-profile"))
                .is_some()
        );
        assert!(
            registry_bytes_for_directory(bytes, std::path::Path::new("another-profile")).is_none()
        );
        assert!(
            registry_bytes_for_directory(
                b"broken".to_vec(),
                std::path::Path::new("synthetic-profile")
            )
            .is_none()
        );
    }
    use super::*;
    fn model() -> LocalLlm {
        LocalLlm {
            token: "a".repeat(64),
            request: None,
            job: None,
            started: Instant::now(),
            transport: None,
            indices: Vec::new(),
        }
    }
    fn candidates() -> Vec<Candidate> {
        ["橋", "箸", "端"]
            .iter()
            .enumerate()
            .map(|(i, s)| Candidate {
                surface: (*s).into(),
                score: -(i as i64),
                segments: 1,
            })
            .collect()
    }
    #[test]
    fn only_existing_candidate_can_be_promoted() {
        let original = candidates();
        let mut actual = original.clone();
        for index in [0, 3, usize::MAX] {
            assert!(!LocalLlm::select(index, &mut actual));
            assert_eq!(actual, original);
        }
        assert!(LocalLlm::select(1, &mut actual));
        assert_eq!(actual[0].surface, "箸");
        assert_eq!(actual.len(), original.len());
        assert!(actual[0].score > actual[1].score);
    }
    #[test]
    fn cancellation_and_empty_context_do_not_contact_backend() {
        let mut llm = model();
        let mut candidates = candidates();
        llm.begin("食事です。", "はし", &candidates);
        assert!(llm.request.is_some());
        llm.job = Some("b".repeat(32));
        llm.cancel();
        assert!(!llm.poll(&mut candidates));
        assert!(llm.request.is_none() && llm.job.is_none());
        llm.begin("", "はし", &candidates);
        assert!(llm.request.is_none());
        llm.begin("食事です。", "はし", &candidates[..1]);
        assert!(llm.request.is_none());
    }

    #[test]
    fn mixed_names_and_numbers_are_preserved_in_model_choices() {
        let mut llm = model();
        let choices: Vec<_> = [
            "ChatGPTでVSCodeの2026年版を使う",
            "ChatGPTでVSCodeの2026年版を遣う",
            "chatgptでVSCodeの2026年版を使う",
            "ChatGPTでVisualStudioの2026年版を使う",
            "ChatGPTでVSCodeの2027年版を使う",
            "ChatGPTでVSCodeの2026年版を使うAI",
            "ChatGPTでVSCodeの2026ねんばんをつかう",
        ]
        .iter()
        .map(|s| Candidate {
            surface: (*s).into(),
            score: 0,
            segments: 1,
        })
        .collect();
        llm.begin(
            "ソフトウェアの話です。",
            "ChatGPTでVSCodeの2026ねんばんをつかう",
            &choices,
        );
        assert_eq!(llm.indices, [0, 1, 2]);
        assert_eq!(
            llm.request.as_ref().unwrap()["candidates"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        // Kana aliases can still acquire an English dictionary spelling.
        let aliases = vec![
            Candidate {
                surface: "AI".into(),
                score: 0,
                segments: 1,
            },
            Candidate {
                surface: "愛".into(),
                score: 0,
                segments: 1,
            },
        ];
        llm.begin("人工知能の話です。", "あい", &aliases);
        assert_eq!(llm.indices, [0, 1]);
        let casing = vec![
            Candidate {
                surface: "chatgpt".into(),
                score: 0,
                segments: 1,
            },
            Candidate {
                surface: "ChatGPT".into(),
                score: -1,
                segments: 1,
            },
        ];
        llm.begin("", "chatgpt", &casing);
        assert_eq!(llm.indices, [0, 1]);
        assert!(llm.pending());
    }

    #[test]
    fn oversized_payload_falls_back_without_pending_inference() {
        let mut llm = model();
        for (context, reading) in [
            ("文".repeat(97), "はし".into()),
            ("文".into(), "読".repeat(129)),
        ] {
            llm.begin(&context, &reading, &candidates());
            assert!(!llm.pending());
        }
    }
}
