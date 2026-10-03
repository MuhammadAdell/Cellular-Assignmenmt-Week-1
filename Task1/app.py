import datetime as dt
import json
import os

import pandas as pd
import streamlit as st

from imagecaption import get_caption
from classifier import ToxicClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, 'database.csv')
DATASET_FILE = os.path.join(HERE, '..', 'cellula toxic data.csv')
DB_COLUMNS = ['timestamp', 'source', 'query', 'image_caption',
              'model', 'predicted_label', 'confidence']

MODEL_DIRS = {
    'BERT + LSTM (LoRA)': os.path.join(HERE, 'models', 'bert_lstm'),
    'BERT + RNN (LoRA)': os.path.join(HERE, 'models', 'bert_rnn'),
}


@st.cache_resource(show_spinner='Loading classification models...')
def load_classifiers():
    loaded = {}
    for name, path in MODEL_DIRS.items():
        if os.path.exists(os.path.join(path, 'metadata.json')):
            loaded[name] = ToxicClassifier(path)
    return loaded


def append_to_database(row):
    file_exists = os.path.exists(DB_FILE)
    pd.DataFrame([row], columns=DB_COLUMNS).to_csv(
        DB_FILE, mode='a', header=not file_exists, index=False)


def main():
    st.set_page_config(page_title='Cellula Toxic Content Classifier',
                       page_icon='🛡️', layout='wide')
    st.title('🛡️ Cellula — Toxic Content Classification')
    st.caption('Muhammad Adel — Task 1: BLIP image captioning + '
               'LoRA-fine-tuned DistilBERT with LSTM / RNN heads')

    classifiers = load_classifiers()
    if not classifiers:
        st.error('No trained models found. Run `python3 train_models.py` first.')
        return

    tab_classify, tab_results, tab_database = st.tabs(
        ['🔍 Classify', '📊 Model results', '📋 View database'])

    # ---------------------------------------------------------- classify tab
    with tab_classify:
        model_name = st.radio('Classification model', list(classifiers))

        input_mode = st.radio('Input type', ['Text only', 'Image only',
                                             'Text + image'], horizontal=True)

        query = ''
        caption = ''
        if input_mode in ('Text only', 'Text + image'):
            query = st.text_area('User text input',
                                 placeholder='Type the text to classify...')
        uploaded = None
        if input_mode in ('Image only', 'Text + image'):
            uploaded = st.file_uploader('Upload an image',
                                        type=['png', 'jpg', 'jpeg', 'webp', 'bmp'])
            if uploaded is not None:
                st.image(uploaded, caption='Uploaded image', width=250)

        if st.button('Classify', type='primary'):
            if input_mode == 'Text only' and not query.strip():
                st.warning('Please enter some text.')
                return
            if input_mode == 'Image only' and uploaded is None:
                st.warning('Please upload an image.')
                return
            if input_mode == 'Text + image' and not query.strip() and uploaded is None:
                st.warning('Please provide text and/or an image.')
                return

            if uploaded is not None:
                from PIL import Image
                with st.spinner('Generating image caption with BLIP...'):
                    caption = get_caption(Image.open(uploaded))
                st.info(f'Generated image caption: **{caption}**')

            clf = classifiers[model_name]
            with st.spinner('Classifying...'):
                if query.strip() and caption:
                    label, conf, probs = clf.predict_pair(query, caption)
                    source = 'text + image caption'
                    text_shown = f'{query} [SEP] {caption}'
                elif query.strip():
                    label, conf, probs = clf.predict(
                        f'{query.strip().lower()} [SEP]')
                    source = 'text'
                    text_shown = query
                else:
                    label, conf, probs = clf.predict(
                        f'[SEP] {caption.strip().lower()}')
                    source = 'image caption'
                    text_shown = caption

            st.success(f'**Prediction: {label}**  (confidence {conf:.1%})')
            st.bar_chart(pd.Series(probs).sort_values(ascending=False))

            append_to_database({
                'timestamp': dt.datetime.now().isoformat(timespec='seconds'),
                'source': source,
                'query': query,
                'image_caption': caption,
                'model': model_name,
                'predicted_label': label,
                'confidence': round(conf, 4),
            })
            st.caption('Record saved to database.csv ✔')

    # --------------------------------------------------------- results tab
    with tab_results:
        st.subheader('Training dataset')
        st.caption(
            'Both models were trained on **cellula toxic data.csv** '
            '(query + image-description pairs). All 9 classes are kept: '
            'exact duplicates and conflicting label pairs are removed, then '
            'rare classes are grown with new, non-redundant query + '
            'description combinations, followed by stratified splitting and '
            'capped oversampling of the training split only.'
        )
        if os.path.exists(DATASET_FILE):
            raw = pd.read_csv(DATASET_FILE)
            kept = None
            for path in MODEL_DIRS.values():
                meta_file = os.path.join(path, 'metadata.json')
                if os.path.exists(meta_file):
                    with open(meta_file) as f:
                        kept = json.load(f).get('dataset_rows_after_cleaning')
                    if kept:
                        break
            c1, c2, c3 = st.columns(3)
            c1.metric('Rows in CSV', f'{len(raw):,}')
            c2.metric('Classes', raw['Toxic Category'].nunique())
            c3.metric('Kept after cleaning', f'{kept:,}' if kept else 'n/a')

        st.divider()
        for name, path in MODEL_DIRS.items():
            meta_file = os.path.join(path, 'metadata.json')
            if not os.path.exists(meta_file):
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            if 'test_per_class' not in meta:
                continue

            st.subheader(name)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Test macro F1', f"{meta['test_macro_f1']:.3f}")
            m2.metric('Test accuracy', f"{meta['test_accuracy']:.1%}")
            m3.metric('Test loss', f"{meta['test_loss']:.4f}")
            m4.metric('Test rows', f"{meta['split_sizes']['test']}")

            per_class = pd.DataFrame(meta['test_per_class']).T.drop(index='accuracy', errors='ignore')
            st.dataframe(per_class.style.format('{:.2f}'), use_container_width=True)

            cm = pd.DataFrame(meta['confusion_matrix'],
                              index=meta['classes'], columns=meta['classes'])
            st.caption('Confusion matrix (rows = true label, columns = predicted label)')
            st.dataframe(cm, use_container_width=True)
            st.divider()

    # -------------------------------------------------------- database tab
    with tab_database:
        st.subheader('All stored inputs and classifications')
        if os.path.exists(DB_FILE):
            db = pd.read_csv(DB_FILE)
            st.dataframe(db, use_container_width=True)
            st.download_button('Download database.csv',
                               pd.read_csv(DB_FILE, encoding='utf-8').to_csv(index=False),
                               file_name='database.csv', mime='text/csv')
        else:
            st.info('The database is empty — classify something first.')

        st.divider()
        st.subheader('Training dataset')
        if os.path.exists(DATASET_FILE):
            with open(DATASET_FILE, 'rb') as f:
                st.download_button('Download cellula toxic data.csv',
                                   data=f.read(),
                                   file_name='cellula toxic data.csv',
                                   mime='text/csv')
        else:
            st.warning('cellula toxic data.csv was not found next to the project folder.')


if __name__ == '__main__':
    main()
