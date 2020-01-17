import torch
from model_class import myBertModel, myTorchDecoder, BertDecoderModel
from flask import Flask, request, render_template
import bs4
import requests
import transformers

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.BertTokenizer.from_pretrained('./model/tokenizer')

PAD_INDEX = 0
BOS_INDEX = 0
EOS_INDEX = 0

encoder_config = transformers.DistilBertConfig.from_pretrained('./model/encoder_config')
encoder = transformers.DistilBertModel(encoder_config)

bert_encoder = myBertModel(encoder, PAD_INDEX)
torch_decoder = myTorchDecoder(bert_encoder.get_embedding(), padding_idx=PAD_INDEX)

vocab_size_out, emb_size_decoder = torch_decoder.get_embedding_dim()
model = BertDecoderModel(bert_encoder, torch_decoder, emb_size_decoder, vocab_size_out).to(device)
model_dict = torch.load('./model/model.pt', map_location=device)
model_dict
model.load_state_dict(model_dict)
model.eval()

del encoder

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    text = request.form['text']
    abstract_text = ''
    title_text = ''
    try:
        page = requests.get(text)
        if page.ok:
            soup = bs4.BeautifulSoup(page.content, 'html.parser')
            soup_title = soup.find('h1', {'class':'title mathjax'})
            soup_abstract = soup.find('blockquote', {'class':'abstract mathjax'})

            title_text = soup_title.text.strip()
            title_text = title_text[len('Title:'):] if title_text.startswith('Title:') else title_text
            title_text = title_text.strip()

            abstract_text = soup_abstract.text.strip()
            abstract_text = abstract_text[len('Abstract:'):] if abstract_text.startswith('Abstract:') else abstract_text
            abstract_text = abstract_text.replace('\n', ' ')
            abstract_text = abstract_text.strip()

            processed_text, _ = translate_sentence(abstract_text)
            processed_text = ' '.join(processed_text).replace('<unk>', '')
            processed_text = processed_text[0].upper() + processed_text[1:]
        else:
            processed_text = f'Wrong URL\nRepsonse: {page.status_code}'
    except Exception as e:
        print(e)
        processed_text = f'Wrong URL:\n{text}'
    return render_template('prediction.html', processed_text=processed_text, url=text, abstract=abstract_text, title_orig=title_text)