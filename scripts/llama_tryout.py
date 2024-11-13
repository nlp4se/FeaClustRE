from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def generate_dynamic_label(cluster_labels):
    unique_labels = list(set(cluster_labels))

    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # Few-shot prompt with examples
    input_text = (
            "Generate a single concise label summarizing the following actions.\n\n"
            "Examples:\n"
            "Video meeting, online meeting, team video chat, conference call\n"
            "Label: Virtual Team Communication\n\n"
            "Secure chat, encrypted messaging, private message\n"
            "Label: Private Messaging\n\n"
            "Video call, group video call, secure video call, video conference\n"
            "Label: Secure Video Conferencing\n\n"
            + ", ".join(unique_labels) + "\n"
                                         "Label:"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    response = pipe(input_text, max_new_tokens=10, do_sample=False, top_k=1)

    label = response[0]['generated_text'].replace(input_text, "").strip()

    return label.split('\n')[0]


cluster_labels = [
"Real-Time Collaboration",
        "Enhanced User Experience",
        "Personalized Recommendations",
        "Instant Notifications",
        "Secure Data Storage",
        "Adaptive Interface",
        "Seamless Integration",
        "Offline Accessibility",
        "Smart Search Functionality",
        "Customizable Dashboards",
        "Interactive Analytics",
        "Voice-Activated Commands",
        "Efficient Task Management",
        "Data Encryption",
        "Multi-Device Sync",
        "In-App Messaging",
        "AI-Powered Suggestions",
        "Intuitive Navigation",
        "Live Support Chat",
        "Advanced Privacy Controls"
    ]
label = generate_dynamic_label(cluster_labels)
print(label)
