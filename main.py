import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, Word2Vec
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Carregar o modelo do spaCy
nlp = spacy.load('pt_core_news_sm')

# Função para extrair tópicos do texto usando spaCy
def extract_topics(text):
    doc = nlp(text)
    topics = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    return ' '.join(topics)

# Dados de exemplo
dados_redes_sociais = {
    'texto': ['Eu amo esse produto!', 'Detestei a experiência de compra.', 'Ótimo atendimento ao cliente!'],
    'sentimento': ['positivo', 'negativo', 'positivo']
}

# Criação do DataFrame do pandas
df_redes_sociais = pd.DataFrame(dados_redes_sociais)

# Feature Engineering: Extração de tópicos usando spaCy
df_redes_sociais['topicos'] = df_redes_sociais['texto'].apply(extract_topics)

# Tratamento de dados desbalanceados com oversampling
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(df_redes_sociais['texto'].values.reshape(-1, 1), df_redes_sociais['sentimento'])

# Pré-processamento com scikit-learn
stop_words = set(stopwords.words('portuguese'))
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(X.flatten())

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento e avaliação de um modelo de Regressão Logística com scikit-learn
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
print('Métricas de desempenho (Regressão Logística - scikit-learn):')
print('Acurácia:', accuracy_lr)
print('Precisão:', precision_lr)
print('Recall:', recall_lr)
print('F1-score:', f1_lr)

# Criação do DataFrame do PySpark
spark = SparkSession.builder.getOrCreate()
spark_df = spark.createDataFrame(df_redes_sociais)

# Pré-processamento com PySpark
string_indexer = StringIndexer(inputCol='sentimento', outputCol='sentimento_index')
spark_df = string_indexer.fit(spark_df).transform(spark_df)

# Tratamento de dados desbalanceados com oversampling com PySpark
spark_df = spark_df.sampleBy('sentimento', fractions={'positivo': 1.0, 'negativo': 1.0}, seed=42, replace=True)

# Divisão dos dados em treinamento e teste com PySpark
spark_train_df, spark_test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# Treinamento e avaliação de um modelo de Regressão Logística com PySpark
word2vec = Word2Vec(inputCol="texto", outputCol="texto_embed")
word2vec_model = word2vec.fit(spark_train_df)
spark_train_df = word2vec_model.transform(spark_train_df)
spark_test_df = word2vec_model.transform(spark_test_df)

spark_model_lr = SparkLogisticRegression(featuresCol='texto_embed', labelCol='sentimento_index')
spark_model_lr = spark_model_lr.fit(spark_train_df)
spark_pred_df_lr = spark_model_lr.transform(spark_test_df)
evaluator_lr = MulticlassClassificationEvaluator(labelCol='sentimento_index', predictionCol='prediction', metricName='accuracy')
accuracy_lr = evaluator_lr.evaluate(spark_pred_df_lr)
print('Acurácia (Regressão Logística - PySpark):', accuracy_lr)

# Treinamento e avaliação de um modelo Random Forest com PySpark
spark_model_rf = SparkRandomForestClassifier(featuresCol='texto_embed', labelCol='sentimento_index')
spark_model_rf = spark_model_rf.fit(spark_train_df)
spark_pred_df_rf = spark_model_rf.transform(spark_test_df)
evaluator_rf = MulticlassClassificationEvaluator(labelCol='sentimento_index', predictionCol='prediction', metricName='accuracy')
accuracy_rf = evaluator_rf.evaluate(spark_pred_df_rf)
print('Acurácia (Random Forest - PySpark):', accuracy_rf)

# Carregamento dos dados pré-processados em um banco de dados usando SQLAlchemy
engine = create_engine('sqlite:///dados_analisados.db')
df_redes_sociais.to_sql('dados_analisados', engine, index=False, if_exists='replace')

# Análise de erros - Regressão Logística
y_pred_labels_lr = [spark_pred_df_lr.select('prediction').collect()[i][0] for i in range(spark_pred_df_lr.count())]
y_true_labels_lr = [spark_pred_df_lr.select('sentimento_index').collect()[i][0] for i in range(spark_pred_df_lr.count())]
error_df_lr = pd.DataFrame({'Texto': spark_pred_df_lr.select('texto').toPandas()['texto'], 'True': y_true_labels_lr, 'Predicted': y_pred_labels_lr})

# Análise de erros - Random Forest
y_pred_labels_rf = [spark_pred_df_rf.select('prediction').collect()[i][0] for i in range(spark_pred_df_rf.count())]
y_true_labels_rf = [spark_pred_df_rf.select('sentimento_index').collect()[i][0] for i in range(spark_pred_df_rf.count())]
error_df_rf = pd.DataFrame({'Texto': spark_pred_df_rf.select('texto').toPandas()['texto'], 'True': y_true_labels_rf, 'Predicted': y_pred_labels_rf})

# Exibição dos resultados
print('Dados pré-processados:')
print(df_redes_sociais)
print('\nAnálise de erros - Regressão Logística:')
print(error_df_lr)
print('\nAnálise de erros - Random Forest:')
print(error_df_rf)

# Visualização dos resultados
metrics = ['Acurácia (Regressão Logística)', 'Acurácia (Random Forest)']
values = [accuracy_lr, accuracy_rf]

plt.figure(figsize=(8, 4))
plt.bar(metrics, values)
plt.title('Métricas de Desempenho')
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.show()
