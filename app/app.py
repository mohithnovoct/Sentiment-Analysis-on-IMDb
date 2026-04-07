import torch
import torch.nn.functional as F
import gradio as gr
import re
from transformers import DistilBertTokenizer

import sys
sys.path.append('..')
from model.classifier import MentalHealthClassifier


print('Loading model...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = MentalHealthClassifier(num_classes=3)

checkpoint = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
model.eval()
print('Model loaded!')

LABEL_NAMES = ['No Distress', 'Mild Distress', 'Severe Distress']
LABEL_COLORS = ['#27AE60', '#F39C12', '#E74C3C']

def predict(text):
    if not text or len(text.strip()) < 5:
        return 'Please enter a longer text (at least a few words).', None

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
    text = text.lower().strip()

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )

    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs  = F.softmax(logits, dim=1)[0]

    predicted_class = probs.argmax().item()
    confidence = probs[predicted_class].item() * 100

    label  = LABEL_NAMES[predicted_class]
    result = f'{label}  ({confidence:.1f}% confidence)'

    # Confidence bar chart for all 3 classes
    prob_dict = {
        f'{LABEL_NAMES[i]}': float(probs[i])
        for i in range(3)
    }

    # Add disclaimer for severe cases
    disclaimer = ''
    if predicted_class == 2:
        disclaimer = ('\n\nNote: This tool is for research purposes only. '
                      'If you or someone you know is in distress, please contact '
                      'a mental health professional or helpline immediately.')

    return result + disclaimer, prob_dict

with gr.Blocks(title='Mental Health Monitor', theme=gr.themes.Soft()) as demo:

    gr.Markdown('## Mental Health Monitoring System')
    gr.Markdown('Analyzes social media text for signs of mental distress.')
    

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label='Enter a post or message:',
                placeholder='Type or paste text here...',
                lines=5
            )
            submit_btn = gr.Button('Analyze', variant='primary')

        with gr.Column():
            result_output = gr.Textbox(label='Prediction')
            prob_output   = gr.Label(label='Confidence Scores')

    # Example inputs for demo
    gr.Examples(
        examples=[
            ['I had such a great day today! Met some old friends and feeling grateful.'],
            ['Work has been really stressful lately. Having trouble sleeping.'],
            ['I feel completely hopeless. Nothing seems to matter anymore.'],
        ],
        inputs=text_input
    )

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[result_output, prob_output]
    )

if __name__ == '__main__':
    demo.launch(share=True) 