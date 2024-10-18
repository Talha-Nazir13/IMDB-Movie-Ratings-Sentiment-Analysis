<h1>IMDB Movie Ratings Sentiment Analysis</h1>

<p>This project focuses on performing sentiment analysis on movie reviews from the IMDB dataset. The goal is to classify reviews as positive or negative based on the textual data using machine learning techniques.</p>

<h2>Features</h2>

<ul>
  <li>Data preprocessing and cleaning of IMDB movie reviews.</li>
  <li>Vectorization of text using <strong>CountVectorizer</strong> and <strong>TF-IDF</strong> (Term Frequency-Inverse Document Frequency).</li>
  <li>Implementation of machine learning models like <strong>Naive Bayes</strong> and <strong>Logistic Regression</strong>.</li>
  <li>Evaluation of model performance using metrics like accuracy, precision, and recall.</li>
  <li>Visualizations of model results using Matplotlib and Seaborn.</li>
</ul>

<h2>Dataset</h2>

<p>The project uses the <strong>IMDB movie review dataset</strong>, which contains movie reviews and their corresponding sentiment labels (positive or negative). The dataset is split into training and testing sets to evaluate model performance.</p>

<h2>Requirements</h2>

<p>To run this project, you will need the following Python libraries:</p>

<ul>
  <li><code>numpy</code></li>
  <li><code>pandas</code></li>
  <li><code>seaborn</code></li>
  <li><code>matplotlib</code></li>
  <li><code>scikit-learn</code> (for text vectorization and model training)</li>
</ul>

<p>Install the dependencies using:</p>

<pre><code>pip install -r requirements.txt
</code></pre>

<h2>Model Architecture</h2>

<p>This project implements two primary models for sentiment classification:</p>

<ul>
  <li><strong>Naive Bayes Classifier</strong>: A simple and fast probabilistic model that works well with text data.</li>
  <li><strong>Logistic Regression</strong>: A linear model used for binary classification tasks, especially suitable for sentiment analysis.</li>
</ul>

<h2>Training</h2>

<p>The models are trained on vectorized text data (using CountVectorizer or TF-IDF). The performance of the models is evaluated on a test set using metrics like accuracy, precision, recall, and F1-score.</p>

<h2>Results</h2>

<p>The project evaluates the models’ performance on the test set, providing detailed classification metrics and visualizations of confusion matrices. The goal is to accurately classify IMDB reviews as positive or negative.</p>

<h2>Customization</h2>

<p>You can adjust the following aspects of the project:</p>

<ul>
  <li><strong>Vectorization technique</strong>: Experiment with other text vectorization methods, such as Word2Vec or BERT embeddings.</li>
  <li><strong>Model choice</strong>: Test other machine learning models like Support Vector Machines (SVM) or deep learning models (e.g., LSTM).</li>
  <li><strong>Hyperparameters</strong>: Tune model hyperparameters to improve performance.</li>
</ul>

<h2>Acknowledgments</h2>

<ul>
  <li>Thanks to the creators of the <strong>IMDB movie review dataset</strong> for providing the data for this project.</li>
  <li>Thanks to open-source libraries like <strong>Scikit-learn</strong>, <strong>Matplotlib</strong>, and <strong>Seaborn</strong> for their powerful tools in data analysis and visualization.</li>
</ul>

