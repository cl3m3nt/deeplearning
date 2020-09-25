import tensorflow as tf 
import tensorflow_recommenders as tfrs 
import tensorflow_datasets as tfds 
import numpy as np 
from typing import Dict,Text

# Get data
ratings = tfds.load('movie_lens/100k-ratings',split='train')
movies = tfds.load('movie_lens/100k-movies',split='train')

# Feature Selection
ratings = ratings.map(lambda x: {
    "movie_title":x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x:x["movie_title"])

# Build vocabularies
user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup()
user_ids_vocabulary.adapt(ratings.map(lambda x:x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup()
movie_titles_vocabulary.adapt(movies)

# Define the movieLens object: a user model, a movie model, a task, a loss function
# Remark: this is quite typical to create a model object when multiple model, like for GAN or VAE(Encoder,Decoder)
class MovieLensModel(tfrs.Model):
    def __init__(self,
        user_model:tf.keras.Model,
        movie_model:tf.keras.Model,
        task: tfrs.tasks.Retrieval):
        super().__init__()

        # user and movie model
        self.user_model = user_model
        self.movie_model = movie_model

        # retrieval task
        self.task = task

    def compute_loss(self,features:Dict[Text,tf.Tensor],training=False)->tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])

        return self.task(user_embeddings,movie_embeddings)


# User & Movie Models, Retrieval task
user_model = tf.keras.models.Sequential([
    user_ids_vocabulary,
    # embeddding layer with input dim and output dim
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(),64)

])

movie_model = tf.keras.models.Sequential([
    movie_titles_vocabulary,
    # embedding layer with input dim and output dim
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(),64)
])

task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
))

# Train and evaluate the model

movie_lens = MovieLensModel(user_model,movie_model,task)
movie_lens.compile(
    optimizer=tf.keras.optimizers.Adagrad(0.5)
)

movie_lens.fit(ratings.batch(4096),epochs=3)

# Use Brute force for retrieval using trained representation
index = tfrs.layers.ann.BruteForce(movie_lens.user_model)
index.index(movies.batch(100).map(movie_lens.movie_model), movies)

# Get some recommendations
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")
