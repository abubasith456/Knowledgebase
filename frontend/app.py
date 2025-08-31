import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000"

def query_api(query):
    try:
        response = requests.post(f"{BACKEND_URL}/api/query", json={"query": query, "n_results": 5})
        if response.status_code == 200:
            return f"Response: {response.json()}"
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=query_api,
    inputs=gr.Textbox(label="Query"),
    outputs=gr.Textbox(label="Response"),
    title="Knowledge Base Frontend"
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
