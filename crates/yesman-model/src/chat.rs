//! ChatML prompt rendering, as used by Qwen2.5-Instruct.

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";

pub struct Message {
    pub role: String,
    pub content: String,
}

pub fn system(content: impl Into<String>) -> Message {
    Message { role: "system".into(), content: content.into() }
}

pub fn user(content: impl Into<String>) -> Message {
    Message { role: "user".into(), content: content.into() }
}

pub fn assistant(content: impl Into<String>) -> Message {
    Message { role: "assistant".into(), content: content.into() }
}

pub const DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// Render a conversation and leave the prompt open at `<|im_start|>assistant`.
pub fn render(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(IM_START);
        out.push('\n');
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str(IM_END);
        out.push('\n');
    }
    out.push_str(IM_START);
    out.push('\n');
    out.push_str("assistant");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_chatml() {
        let s = render(&[system("hi"), user("yo")]);
        assert_eq!(
            s,
            "<|im_start|>\nsystem\nhi<|im_end|>\n<|im_start|>\nuser\nyo<|im_end|>\n<|im_start|>\nassistant\n"
        );
    }
}
