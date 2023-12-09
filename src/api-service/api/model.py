import os
import re
import json
import numpy as np
import tensorflow as tf
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import tensorflow_hub as hub
from google.cloud import aiplatform
import base64
from transformers import BertTokenizer

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize.toktok import ToktokTokenizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# AUTOTUNE = tf.data.experimental.AUTOTUNE
local_experiments_path = "/persistent/experiments"


# best_model = None
# best_model_id = None
# prediction_model = None
# data_details = None


# Replace 'bucket-name' and 'object-path' with your GCS bucket and object path

def download_gpc_model(bucket_name):
    client = storage.Client()

    print(bucket_name)
    bucket_name = bucket_name.replace("gs://", "")
    source_dir = "v1/model"

    # Specify the local destination path
    dest_dir = '/app/train_data'
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_dir)

    # Download the GCS object to the local destination
    for blob in blobs:
        blob.download_to_filename(os.path.join(dest_dir, os.path.basename(blob.name)))
        print("Downloaded ", os.path.basename(blob.name))


def load_prediction_model():
    print("Loading Model...")
    global prediction_model, data_details

    best_model_path = "model_structure.json"
    best_weights_path = "model_weights.h5"

    print("best_model_path:", best_model_path)
    print("best_model_weights_path:", best_weights_path)

    # Load the JSON string from the file
    with open(best_model_path, "r") as json_file:
        best_model_json = json_file.read()

    # Reconstruct the model from the JSON string
    best_model = model_from_json(best_model_json)

    best_model.load_weights(best_weights_path)

    learning_rate = 0.001
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.001)
    loss = keras.losses.SparseCategoricalCrossentropy()

    best_model.compile(loss=loss,
                       optimizer=optimizer,
                       metrics=['accuracy'])

    print(best_model.summary())


def load_text_from_path(text_path):
    print("input text: ", text_path)

    input_dim = 10000

    input_text = []

    with open(text_path, 'r') as file:
        for line in file:
            # Add each line to the list, optionally stripping the newline character
            input_text.append(line.strip())

    # processor = TextProcessor()
    processor = BertTextProcessor()

    processed_text = processor.process_text(input_text)
    print("processed text: ", processed_text)

    vectorizer = TfidfVectorizer(max_features=input_dim)
    input_tfidf = vectorizer.fit_transform(processed_text).toarray()

    # processed_text = np.array(processed_text)
    return (input_tfidf)


def make_prediction(text_path):
    # Load & preprocess
    test_data = load_text_from_path(text_path)
    model = load_prediction_model()

    # Make prediction
    prediction = model.predict(test_data)
    idx = prediction.argmax(axis=1)[0]
    prediction_label = data_details["index2label"][str(idx)]

    if prediction_model.layers[-1].activation.__name__ != "softmax":
        prediction = tf.nn.softmax(prediction).numpy()
        print(prediction)

    poisonous = False
    if prediction_label == "amanita":
        poisonous = True

    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction) * 100, 2),
        "poisonous": poisonous,
    }


import emoji
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize.toktok import ToktokTokenizer

class BertTextProcessor:

    def __init__(self):
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def remove_special_characters(self, text):
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)

    def remove_between_square_brackets(self, text):
        return re.sub('\[[^]]*\]', '', text)

    def remove_emoji(self, text):
        return emoji.demojize(text)

    def bert_tokenize(self, text):
        # Tokenize and encode sequences in the input text
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
            max_length = 144,           # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length = True,
            return_attention_mask = True,   # Construct attention masks
            return_tensors = 'pt',     # Return pytorch tensors
        )
        return encoded_dict

    def process_dataframe(self, df, text_column):
        # Preprocess and store in a new column
        df['simple_processed_text'] = (df[text_column]
                                      #  .apply(self.to_lower_case)
                                       .apply(self.remove_special_characters)
                                       .apply(self.remove_between_square_brackets)
                                       .apply(self.remove_emoji))

        # BERT tokenization for model input
        df['input_ids'] = df['simple_processed_text'].apply(lambda x: self.bert_tokenize(x)['input_ids'].numpy()[0])
        df['attention_mask'] = df['simple_processed_text'].apply(lambda x: self.bert_tokenize(x)['attention_mask'].numpy()[0])
        return df
        
    def process_dataframe(self, df, text_column):
        # Preprocess and store in a new column
        df['simple_processed_text'] = (df[text_column]
                                      #  .apply(self.to_lower_case)
                                       .apply(self.remove_special_characters)
                                       .apply(self.remove_between_square_brackets)
                                       .apply(self.remove_emoji))

        # BERT tokenization for model input
        df['input_ids'] = df['simple_processed_text'].apply(lambda x: self.bert_tokenize(x)['input_ids'].numpy()[0])
        df['attention_mask'] = df['simple_processed_text'].apply(lambda x: self.bert_tokenize(x)['attention_mask'].numpy()[0])
        return df
        
    def process_text(self, text_list):
        input_ids_list = []
        attention_masks_list = []
        # Process and tokenize each text in the list
        for text in text_list:
            # Apply preprocessing steps
            processed_text = self.remove_special_characters(text)
            processed_text = self.remove_between_square_brackets(processed_text)
            processed_text = self.remove_emoji(processed_text)
            # Tokenize
            encoded_dict = self.bert_tokenize(processed_text)
            # Extract and store the tokenized values
            input_ids_list.append(encoded_dict['input_ids'].numpy()[0])
            attention_masks_list.append(encoded_dict['attention_mask'].numpy()[0])

        # Convert lists to numpy arrays for compatibility with many ML frameworks
        input_ids_array = np.array(input_ids_list)
        attention_masks_array = np.array(attention_masks_list)

        return input_ids_array, attention_masks_array

# class TextProcessor:

#     def __init__(self):
#         self.tokenizer = ToktokTokenizer()
#         self.lemmatizer = WordNetLemmatizer()
#         self.stopword_list = nltk.corpus.stopwords.words('english')
#         custom_stopword = ['[', ']', '[]']
#         self.stopword_list.extend(custom_stopword)

#     def to_lower_case(self, text):
#         return text.lower()

#     def remove_stopwords(self, text, is_lower_case=False):
#         tokens = self.tokenizer.tokenize(text)
#         tokens = [token.strip() for token in tokens]
#         if is_lower_case:
#             filtered_tokens = [token for token in tokens if token not in self.stopword_list]
#         else:
#             filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list]
#         return ' '.join(filtered_tokens)

#     def remove_special_characters(self, text, remove_digits=True):
#         pattern = r'[^a-zA-z0-9\s]'
#         return re.sub(pattern, '', text)

#     def remove_between_square_brackets(self, text):
#         return re.sub('\[[^]]*\]', '', text)

#     def lemmatize_sentence(self, sentence):
#         word_list = word_tokenize(sentence)
#         return ' '.join([self.lemmatizer.lemmatize(w) for w in word_list])

#     def remove_emoji(self, text):
#         return emoji.demojize(text)

#     def remove_tags(self, text):
#         hashtag_regex = r'#\S+'
#         cleaned_text = re.sub(hashtag_regex, '', text)
#         return cleaned_text.strip()

#     def process_text(self, text_list):
#         print(text_list)
#         processed = [self.to_lower_case(i) for i in text_list]
#         processed = [self.remove_tags(i) for i in processed]
#         processed = [self.remove_stopwords(i) for i in processed]
#         processed = [self.remove_special_characters(i) for i in processed]
#         processed = [self.remove_between_square_brackets(i) for i in processed]
#         processed = [self.lemmatize_sentence(i) for i in processed]
#         processed = [self.remove_emoji(i) for i in processed]
#         processed = [i for i in processed if i != ""]
#         print(processed)

# Tensorflow
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.utils.layer_utils import count_params
#
#
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# local_experiments_path = "/persistent/experiments"
# best_model = None
# best_model_id = None
# prediction_model = None
# data_details = None
# image_width = 224
# image_height = 224
# num_channels = 3
#
#
# def load_prediction_model():
#     print("Loading Model...")
#     global prediction_model, data_details
#
#     best_model_path = os.path.join(
#         local_experiments_path,
#         best_model["experiment"],
#         best_model["model_name"] + ".keras",
#     )
#
#     print("best_model_path:", best_model_path)
#     prediction_model = tf.keras.models.load_model(
#         best_model_path, custom_objects={"KerasLayer": hub.KerasLayer}
#     )
#     print(prediction_model.summary())
#
#     data_details_path = os.path.join(
#         local_experiments_path, best_model["experiment"], "data_details.json"
#     )
#
#     # Load data details
#     with open(data_details_path, "r") as json_file:
#         data_details = json.load(json_file)
#
#
# def check_model_change():
#     global best_model, best_model_id
#     best_model_json = os.path.join(local_experiments_path, "best_model.json")
#     if os.path.exists(best_model_json):
#         with open(best_model_json) as json_file:
#             best_model = json.load(json_file)
#
#         if best_model_id != best_model["experiment"]:
#             load_prediction_model()
#             best_model_id = best_model["experiment"]
#
# def parse_tf_example(example_proto, vector_size, num_classes):
#     feature_description = {
#         'text_vector': tf.io.FixedLenFeature(shape=(vector_size,), dtype=tf.float32),
#         'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
#     }
#     example = tf.io.parse_single_example(example_proto, feature_description)
#     label = tf.one_hot(example['label'], num_classes)
#     return example['text_vector'], label
# def load_tfrecords_dataset(filename, vector_size, num_classes, shuffle_buffer_size=None):
#
#     dataset = tf.data.TFRecordDataset(filename)
#     if shuffle_buffer_size:
#         dataset = dataset.shuffle(shuffle_buffer_size)
#     dataset = dataset.map(lambda x: parse_tf_example(x, vector_size, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(1)
#     dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#     return dataset
#
# # def load_preprocess_image_from_path(image_path):
# #     print("Image", image_path)
# #
# #     image_width = 224
# #     image_height = 224
# #     num_channels = 3
# #
# #     # Prepare the data
# #     def load_image(path):
# #         image = tf.io.read_file(path)
# #         image = tf.image.decode_jpeg(image, channels=num_channels)
# #         image = tf.image.resize(image, [image_height, image_width])
# #         return image
# #
# #     # Normalize pixels
# #     def normalize(image):
# #         image = image / 255
# #         return image
# #
# #     test_data = tf.data.Dataset.from_tensor_slices(([image_path]))
# #     test_data = test_data.map(load_image, num_parallel_calls=AUTOTUNE)
# #     test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
# #     test_data = test_data.repeat(1).batch(1)
# #
# #     return test_data
#
#
# def make_prediction(image_path):
#     check_model_change()
#
#     # Load & preprocess
#     filename = os.path.join(image_path, )
#     test_data = load_tfrecords_dataset(filename)
#     # test_data = load_preprocess_image_from_path(image_path)
#
#     # Make prediction
#     prediction = prediction_model.predict(test_data)
#     idx = prediction.argmax(axis=1)[0]
#     prediction_label = data_details["index2label"][str(idx)]
#
#     if prediction_model.layers[-1].activation.__name__ != "softmax":
#         prediction = tf.nn.softmax(prediction).numpy()
#         print(prediction)
#
#     poisonous = False
#     if prediction_label == "amanita":
#         poisonous = True
#
#     return {
#         "input_image_shape": str(test_data.element_spec.shape),
#         "prediction_shape": prediction.shape,
#         "prediction_label": prediction_label,
#         "prediction": prediction.tolist(),
#         "accuracy": round(np.max(prediction) * 100, 2),
#         "poisonous": poisonous,
#     }
#
#
# # def make_prediction_vertexai(image_path):
# #     print("Predict using Vertex AI endpoint")
# #
# #     # Get the endpoint
# #     # Endpoint format: endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}"
# #     endpoint = aiplatform.Endpoint(
# #         "projects/129349313346/locations/us-central1/endpoints/8600804363952193536"
# #     )
# #
# #     with open(image_path, "rb") as f:
# #         data = f.read()
# #     b64str = base64.b64encode(data).decode("utf-8")
# #     # The format of each instance should conform to the deployed model's prediction input schema.
# #     instances = [{"bytes_inputs": {"b64": b64str}}]
# #
# #     result = endpoint.predict(instances=instances)
# #
# #     print("Result:", result)
# #     prediction = result.predictions[0]
# #     print(prediction, prediction.index(max(prediction)))
# #
# #     index2label = {0: "oyster", 1: "crimini", 2: "amanita"}
# #
# #     prediction_label = index2label[prediction.index(max(prediction))]
# #
# #     poisonous = False
# #     if prediction_label == "amanita":
# #         poisonous = True
# #
# #     return {
# #         "prediction_label": prediction_label,
# #         "prediction": prediction,
# #         "accuracy": round(np.max(prediction) * 100, 2),
# #         "poisonous": poisonous,
# #     }
