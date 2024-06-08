# -textclassification-nlp

**The code begins by importing the necessary libraries for working with TensorFlow, TensorFlow Hub, TensorFlow Text, Pandas, NumPy, and scikit-learn.

The dataset is loaded from a CSV file (spam.csv) using the read_csv function from Pandas. The DataFrame df is created to hold the dataset.

The head(5) function is used to display the first 5 rows of the DataFrame, giving a glimpse of the dataset.

The groupby function is applied to the DataFrame to perform descriptive statistics on the 'Category' column. This provides insights into the distribution of categories in the dataset.

The value_counts function is used to count the number of occurrences of each category in the 'Category' column. This helps in understanding the class distribution.

The DataFrame is filtered to create a new DataFrame df_spam that contains only the rows where the 'Category' is 'spam'. The .shape attribute is used to display the dimensions of the DataFrame, indicating the number of rows and columns.

Similarly, another DataFrame df_ham is created by filtering the original DataFrame for rows where the 'Category' is 'ham'. Again, the .shape attribute is used to display the dimensions.

To balance the dataset, the 'ham' DataFrame is downsampled to have the same number of rows as the 'spam' DataFrame. The .sample function is used to randomly select rows from the 'ham' DataFrame. The resulting downsampled DataFrame is stored in df_ham_downsampled, and its dimensions are displayed using .shape.

The downsampled 'ham' DataFrame and the 'spam' DataFrame are concatenated using pd.concat to create a balanced DataFrame df_balanced. The .shape attribute is used to display the dimensions of the balanced DataFrame.

The balanced DataFrame is further analyzed by counting the number of occurrences of each category in the 'Category' column using .value_counts(). This verifies if the balancing operation was successful.

A new column named 'spam' is added to the balanced DataFrame. The 'spam' column is created using the apply function, which maps the value 'spam' to 1 and 'ham' to 0.

The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The 'Message' column of the balanced DataFrame is used as the feature data (X), and the 'spam' column is used as the target variable (y). The stratify parameter ensures that the class distribution is maintained in both the training and testing sets. The .head(4) function is used to display the first 4 rows of the training set.

The BERT preprocessing and encoding models are loaded using TensorFlow Hub. The hub.KerasLayer is used to wrap the URLs of the BERT models.

The function get_sentence_embedding is defined to preprocess sentences using the BERT preprocessing model and obtain their embeddings using the BERT encoder. The function takes a list of sentences as input and returns the sentence embeddings.

The get_sentence_embedding function is tested by passing a list of example sentences and printing the corresponding sentence embeddings.

The cosine_similarity function from scikit-learn is used to compute the cosine similarity between pairs of sentence embeddings. Several examples are provided to demonstrate the usage of cosine similarity.

The input layer of the neural network model is defined using tf.keras.layers.Input, specifying the shape and data type of the input tensor.

The BERT preprocessing layer is applied to the input layer to preprocess the text data.

The BERT encoder layer is applied to the preprocessed text to obtain the outputs, which include the pooled output representing the sentence embeddings.

Neural network layers are defined to process the sentence embeddings. In this case, a dropout layer with a dropout rate of 0.1 and a dense layer with sigmoid activation are used.

The inputs and outputs are used to construct the final model using tf.keras.Model.

The model summary is printed, displaying the architecture and the number of trainable parameters.

The length of the training set (X_train) is calculated using len.

A list of metrics is defined to evaluate the model's performance during training. The metrics include binary accuracy, precision, and recall.

The model is compiled with the Adam optimizer, binary cross-entropy loss, and the defined metrics.

The model is trained on the training data (X_train and y_train) for a specified number of epochs.

The model is evaluated on the testing data (X_test and y_test) using model.evaluate.

The model's predictions are obtained on the testing data using model.predict. The predictions are flattened to a 1D array.

The predicted values are converted to binary class labels based on a threshold of 0.5.

The confusion matrix is computed using confusion_matrix from scikit-learn to evaluate the model's performance.

The confusion matrix is visualized using a heatmap created with seaborn.heatmap and annotated with the values.

The classification report is generated using classification_report from scikit-learn to provide a comprehensive evaluation of the model's performance, including precision, recall, F1-score, and support for each class.

The classification report is printed, displaying the evaluation metrics for each class.

To increase the accuracy we can increase the number of epochs**
