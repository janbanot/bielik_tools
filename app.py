import gradio as gr
import json
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load prompts from JSON file
with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)


def get_system_message(prompt_id):
    """Get system message for a given prompt ID"""
    for prompt in PROMPTS:
        if prompt["id"] == prompt_id:
            return prompt["system_message"]
    return ""


def transform_text(text, prompt_id):
    """Transform text using the selected prompt"""
    if not text:
        return "Proszę wprowadzić tekst do transformacji"

    try:
        system_message = get_system_message(prompt_id)

        extended_system_message = f"Jesteś ekspertem zajmującym się poprawą tesktów w języku polskim. Zmień otrzymany tekst zgodnie z podaną instrukcją. Pamietaj, aby ingerencja w treść była związana tylko z poprawą błędu i nie ingerowala w sens pierwotnej treści. Zwróć tylko poprawiony tekst bez dodatkowych komentarzy. Instrukcja: {system_message}" # noqa

        output = replicate.run(
            "aleksanderobuchowski/bielik-11b-v2.3-instruct:dc287cda645e8f80c83ccb1b01c8c8fe8d652b4040c073e3c75112f20f983a2a", # noqa
            input={
                "input": text,
                "top_p": 1,
                "max_length": 1000,
                "temperature": 0.75,
                "system_message": extended_system_message,
                "repetition_penalty": 1,
            },
            timeout=120  # Increase timeout to 60 seconds
        )
        return "".join(output)
    except Exception as e:
        return f"Błąd przetwarzania: {str(e)}"


def update_description(prompt_id):
    """Update description when prompt selection changes"""
    for prompt in PROMPTS:
        if prompt["id"] == prompt_id:
            return prompt["description"]
    return ""


# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Transformator Tekstu")

    with gr.Row():
        with gr.Column():
            prompt_dropdown = gr.Dropdown(
                choices=[(p["name"], p["id"]) for p in PROMPTS],
                label="Wybierz typ transformacji",
            )
            description = gr.Textbox(label="Opis transformacji", interactive=False)
            input_text = gr.Textbox(label="Tekst wejściowy", lines=5)

        output_text = gr.Textbox(label="Tekst wynikowy", lines=5, interactive=False)

    submit_btn = gr.Button("Przetwórz tekst")

    prompt_dropdown.change(
        fn=update_description, inputs=prompt_dropdown, outputs=description
    )

    submit_btn.click(
        fn=transform_text, inputs=[input_text, prompt_dropdown], outputs=output_text
    )

if __name__ == "__main__":
    app.launch()
