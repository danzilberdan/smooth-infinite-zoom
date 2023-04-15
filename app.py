import gradio as gr
from zoom import zoom_app
app = gr.Blocks()
with app:
    gr.HTML(
        """
        <p style='text-align: center'>
       Text to Video - Infinite zoom effect
        </p>
        """
    )
    zoom_app()

app.launch(debug=True, enable_queue=True)
