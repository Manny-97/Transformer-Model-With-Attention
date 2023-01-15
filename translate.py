import pickle
import numpy as np
from flask import Flask, jsonify, request, Response

with open('') as fp:
    model = pickle.load(fp)

with open('vectorize.pickle', 'rb') as fp:
    data = pickle.load(fp)

eng_vectorizer = data['eng_vectorizer']
fra_vectorizer = data['fra_vectorizer']
seq_len = data['seq_len']
vocab_size_fr = data['vocab_size_fr']

def translate(sentence):
    """Create the translated sentence"""
    enc_tokens = eng_vectorizer([sentence])
    lookup = list(fra_vectorizer.get_vocabulary())
    start_sentinel, end_sentinel = "[start]", "[end]"
    output_sentence = [start_sentinel]
    # generate the translated sentence word by word
    for i in range(seq_len):
        vector = fra_vectorizer([" ".join(output_sentence)])
        assert vector.shape == (1, seq_len+1)
        dec_tokens = vector[:, :-1]
        assert dec_tokens.shape == (1, seq_len)
        pred = model([enc_tokens, dec_tokens])
        assert pred.shape == (1, seq_len, vocab_size_fr)
        word = lookup[np.argmax(pred[0, i, :])]
        output_sentence.append(word)
        if word == end_sentinel:
            break
    return output_sentence


app = Flask('language-translate')

@app.route('/translate', methods=["POST"])
def predict_endpoint():
    sentence = request.get_json()
    french = translate(sentence)
    result = {
        'French': french
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
