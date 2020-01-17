import torch
import torch.nn.functional as F
from model_class import myBertModel, myTorchDecoder, BertDecoderModel
from flask import Flask, request, render_template
import bs4
import requests
import transformers
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('dark_background')

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.BertTokenizer.from_pretrained('./model/tokenizer')

PAD_INDEX = 0
BOS_INDEX = 101
EOS_INDEX = 102
MAX_OUT_LEN = 50

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

def beam_search_batch(model, generate_len, tensor_in, beam_size, n_prev_tokens_exclude=3, model_extra_params=[]):
    model.eval()
    batch_size = tensor_in.shape[0]
    _min_value = -1000 
    tensor_hyp = torch.ones((batch_size, beam_size, generate_len + 1), dtype=torch.long, device=device) * BOS_INDEX
    logits_sum = torch.ones((batch_size, beam_size), dtype=torch.float32, device=device) * _min_value
    logits_sum[:, 0] = 0
    tensor_in_calc = torch.repeat_interleave(tensor_in, beam_size, 0)
    if len(model_extra_params) == 2:
        model_extra_params_calc = []
        model_extra_params_calc.append(torch.repeat_interleave(model_extra_params[0], beam_size, 0))
        model_extra_params_calc.append(torch.repeat_interleave(model_extra_params[1], beam_size, 0))
    generated_dict = {}
    with torch.no_grad():
        for i in range(generate_len):
            # forward pass
            tensor_out = tensor_hyp[:, :, :i+1].view(batch_size * beam_size, -1)
            logits, *model_extra_params_calc = model(tensor_in_calc, tensor_out, MAX_OUT_LEN, True, *model_extra_params_calc)
            logits = logits[:, -1, :]
            logits = F.log_softmax(logits, -1)
            # exclude recent tokens
            for j in range(n_prev_tokens_exclude):
                if i >= j:
                    logits[:, tensor_out[:, i-j]] = _min_value
            # get top values/indices
            logits_by_batch = logits.view(batch_size, beam_size, -1)
            logits_topk_val, logits_topk_idx = logits_by_batch.topk(beam_size, dim=-1)
            # save strings with EOS token, calc top values again
            for b, h, _ in (logits_topk_idx == EOS_INDEX).nonzero().tolist():
                if i >= 6:
                    if b in generated_dict.keys():
                        generated_dict[b].append((logits_sum[b, h].item(), tensor_hyp[b, h, :i+1].tolist()))
                    else:
                        generated_dict[b] = [(logits_sum[b, h].item(), tensor_hyp[b, h, :i+1].tolist())]
                logits_by_batch[b, h, EOS_INDEX] = _min_value
            logits_topk_val, logits_topk_idx = logits_by_batch.topk(beam_size, dim=-1)
            # add logits
            logits_topk_val_by_batch = logits_topk_val.view(batch_size, -1)
            logits_topk_idx_by_batch = logits_topk_idx.view(batch_size, -1)
            logits_sum_mult = torch.repeat_interleave(logits_sum, beam_size, 1) + logits_topk_val_by_batch
            # select top indices
            logits_sum_topk_val, logits_sum_topk_idx_sub = logits_sum_mult.topk(beam_size, -1)
            logits_sum_topk_idx = logits_topk_idx_by_batch.gather(1, logits_sum_topk_idx_sub)
            tensor_hyp[:, :, i+1] = logits_sum_topk_idx
            logits_sum = logits_sum_topk_val
        for b in range(tensor_hyp.shape[0]):
            for h in range(tensor_hyp.shape[1]):
                if b in generated_dict.keys():
                    generated_dict[b].append((logits_sum[b, h].item(), tensor_hyp[b, h, :i+1].tolist()))
                else:
                    generated_dict[b] = [(logits_sum[b, h].item(), tensor_hyp[b, h, :i+1].tolist())]

    return generated_dict

def display_attention(attention, sentence, translation, filename='foo.png'):
    fig = plt.figure(figsize=(40,50))
    ax = fig.add_subplot(111)

    sentence = sentence[:sentence.index(sentence[-1])]
    attention = attention[:, :len(sentence)].numpy().T
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=10)
    ax.set_yticklabels(['']+sentence)
    ax.set_xticklabels(['']+translation, rotation=80)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(filename)

def generate_title_attention(text):
    text_tensor = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    # forward pass through encoder
    text_encoded_params = model.encoder(text_tensor)
    # beam search
    beam_res_dict = beam_search_batch(model, generate_len=10, tensor_in=text_tensor, beam_size=3, model_extra_params=text_encoded_params)
    beam_res_dict_decoded = [tokenizer.decode(t[1:]) for s, t in beam_res_dict[0]]
    # attention
    beam_top_res = beam_res_dict[0][0][1]
    beam_top_res_tensor = torch.LongTensor(beam_top_res).to(device).unsqueeze(0)
    _, attention_tensors = torch_decoder.forward(beam_top_res_tensor, *text_encoded_params, MAX_OUT_LEN, True)
    text_tokenized = tokenizer.tokenize(text)
    beam_res_tokenized = tokenizer.tokenize(tokenizer.decode(beam_top_res))
    attention_filename = f'static/{hash(text)}.png'
    display_attention(attention_tensors[1][0].data, text_tokenized, beam_res_tokenized, attention_filename)

    return beam_res_dict_decoded, attention_filename

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    text = request.form['text']
    abstract_text = ''
    title_text = ''
    title_hypothesis_2 = ''
    title_hypothesis_3 = ''
    title_hypothesis_4 = ''
    attention_filename = ''
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

            title_hypotheses, attention_filename = generate_title_attention(abstract_text)
            title_hypothesis_1 = string.capwords(title_hypotheses[0]) if len(title_hypotheses) >= 1 else '-'
            title_hypothesis_2 = string.capwords(title_hypotheses[1]) if len(title_hypotheses) >= 2 else '-'
            title_hypothesis_3 = string.capwords(title_hypotheses[2]) if len(title_hypotheses) >= 3 else '-'
            title_hypothesis_4 = string.capwords(title_hypotheses[3]) if len(title_hypotheses) >= 4 else '-'
        else:
            title_hypothesis_1 = f'Wrong URL\nRepsonse: {page.status_code}'
    except Exception as e:
        print(e)
        title_hypothesis_1 = f'Wrong URL:\n{text}'
    return render_template(
        'prediction.html'
        , title_hypothesis_1=title_hypothesis_1
        , title_hypothesis_2=title_hypothesis_2
        , title_hypothesis_3=title_hypothesis_3
        , title_hypothesis_4=title_hypothesis_4
        , attention_filename=attention_filename
        , url=text, abstract=abstract_text, title_orig=title_text
    )