# %%
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk import word_tokenize
import re
import requests
from sentence_transformers import SentenceTransformer, util
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import yake
import streamlit as st


# %%
url = st.text_input("Insert the URL to search for Fever plans.")

if st.button("Search for Fever Plans"):
    if url:
# %%
        planes_vlc = pd.read_csv("planes_vlc.csv")
        plans_chg = pd.read_csv("plans_chg.csv")

        # %%
        plans = plans_chg['DS_PLAN'].tolist()

        # %%
        # Convierte todos los elementos a minúsculas
        plans_l = [text.lower() for text in plans]

        # %%
        # Crea una máscara de traducción para eliminar los signos de puntuación
        translator = str.maketrans('', '', string.punctuation)

        # Aplica la máscara a cada elemento de la lista
        plans_p = [elemento.translate(translator) for elemento in plans_l]

        # Ahora, lista_sin_puntuacion contiene los elementos sin signos de puntuación
        # print(plans_p)

        # %%
        # Lista para almacenar los tokens de cada texto
        plans_tkn = []

        # Tokeniza cada texto y agrega la lista de tokens a lista_de_listas_de_tokens
        for text in plans_p:
            tokens = word_tokenize(text)
            plans_tkn.append(tokens)

        # Ahora, lista_de_listas_de_tokens contiene listas de tokens para cada texto
        # for tokens in plans_tkn:
        #     print(tokens)

        # %%
        stop_words = set(stopwords.words('english')) # Set of stopwords
        # stop_words = set(stopwords.words('spanish')) # Set of stopwords

        # %%
        plans_sw = []

        # Iterar a través de las listas de tokens y aplicar la eliminación de stopwords
        for words in plans_tkn:
            tkn_sw = [word for word in words if word not in stop_words]
            plans_sw.append(tkn_sw)

        # Ahora, lista_de_listas_de_tokens_sin_stopwords contiene listas de tokens sin stopwords
        # for text_sw in plans_sw:
        #     print(text_sw)

        # %%
        # Convierte la lista de listas en una serie de pandas
        plans_sw_serie = pd.Series(plans_sw)

        # Agrega la nueva columna al DataFrame
        plans_chg['tokens'] = plans_sw_serie

        # Imprime el DataFrame resultante
        # print(plans_chg)

        # %%
        # Define las configuraciones del extractor de palabras clave YAKE
        language = "en"
        max_ngram_size = 5
        deduplication_threshold = 0.9
        numOfKeywords = 20
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

        # Lista para almacenar todas las palabras clave de todas las listas
        todas_las_palabras_clave = []

        # Itera a través de las listas de tokens sin stopwords y aplica YAKE
        for index, row in plans_chg.iterrows():
            lista_tokens = row['tokens']
            texto = " ".join(lista_tokens)
            keywords = custom_kw_extractor.extract_keywords(texto)
            
            # Filtra las palabras clave que tienen dos o más palabras (no son unigramas)
            # keywords_sin_unigramas = [kw for kw in keywords if ' ' in kw[0]]

            # Agrega el código ID_PLAN a cada palabra clave generada
            keywords_con_id = [(row['ID_PLAN'], kw[0], kw[1]) for kw in keywords]
            
            # Extiende la lista principal con las palabras clave de esta lista
            todas_las_palabras_clave.extend(keywords_con_id)

        # Imprime todas las palabras clave en una sola lista
        # for kw in todas_las_palabras_clave:
        #     print(f"ID_PLAN: {kw[0]}, Palabra Clave: {kw[1]}, Puntuación: {kw[2]}")


        # Realizar la solicitud HTTP
        get_url = requests.get(url)
        get_text = get_url.text
        soup = BeautifulSoup(get_text, "html.parser")

        # Buscar la sección 'article__body col-md-8'
        all_content = soup.find("section", class_="article__body col-md-8")

        # Buscar todos los elementos <p> y <h2>
        text_elements = all_content.find_all(['p', 'h2'])

        paragraphs = []
        # Expresión regular para detectar emoticonos
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')

        for element in text_elements:
            if element.name == 'p':
                if element.get_text(strip=True) == '':
                    continue
                # Filtrar los párrafos que no contienen imágenes ni emoticonos
                if not element.find('img') and not emoji_pattern.search(element.get_text()):
                    if not element.find_parent('blockquote') and not element.find_parent('iframe', class_='instagram-media instagram-media-rendered'):
                        paragraphs.append(element.get_text())
            elif element.name == 'h2':
                paragraphs.append(element.get_text())

        # Imprimir los párrafos y encabezados <h2>
        # for paragraph in paragraphs:
        #     print(paragraph)

        # %%
        # Lista para almacenar los resultados
        arts = []

        # Iterar sobre cada texto y aplicar el proceso de preprocesamiento
        for text in paragraphs:
        
            # Tokenización
            tokens = sent_tokenize(text)

            def remove_punctuation(sentence):
                return re.sub(r'[^\w\s]', '', sentence)

            # Aplicar la función a todas las frases
            sents = [remove_punctuation(sentence) for sentence in tokens]

            arts.append(sents) 
            

        # Imprimir los resultados procesados
        # for idx, processed_text in enumerate(arts):
        #     print(f"Processed Text {idx+1}: {processed_text}")

        # %%
        def convertir_listas_a_minusculas(listas):
            for lista in listas:
                for i in range(len(lista)):
                    lista[i] = lista[i].lower()

        # Convertir las listas a minúsculas
        if arts:
            convertir_listas_a_minusculas(arts)

            # Imprimir las listas modificadas
            # for lista in arts:
            #     print(lista)

        # %%
        # Función para tokenizar las palabras en todas las listas
        def tokenizar_listas(listas):
            arts_tkn = []
            for lista in arts:
                tokens = [word_tokenize(word) for word in lista]
                arts_tkn.append(tokens)
            return arts_tkn

        # Tokenizar las listas
        if arts:
            arts_tkn = tokenizar_listas(arts)

            # Imprimir las listas tokenizadas
            # for lista in arts_tkn:
            #     print(lista)

        # %%
        # Función para eliminar stopwords de una lista de palabras tokenizadas
        def eliminar_stopwords_de_lista(lista_tokenizada):
            # stop_words = set(stopwords.words("english"))  # Puedes cambiar "english" por otro idioma si es necesario
            lista_sin_stopwords = [palabra for palabra in lista_tokenizada if palabra.lower() not in stop_words]
            return lista_sin_stopwords

        # Función para combinar palabras en una lista en una sola frase
        def combinar_palabras_en_frase(lista_palabras):
            frase = " ".join(lista_palabras)
            return [frase]

        # Procesar la estructura de tres niveles y almacenar los resultados en una nueva lista
        resultado_final = []
        for lista_de_listas in arts_tkn:
            lista_resultado_nivel_1 = []
            for lista in lista_de_listas:
                lista_sin_stopwords = eliminar_stopwords_de_lista(lista)
                frase = combinar_palabras_en_frase(lista_sin_stopwords)
                lista_resultado_nivel_1.extend(frase)
            resultado_final.extend(lista_resultado_nivel_1)

        # Imprimir el resultado final
        # for frase in resultado_final:
        #     print(frase)

        # %%
        # Lista para almacenar resultados
        resu_all = []

        # Recorrer cada keyword de todas_las_palabras_clave
        for id, keyword, puntuacion in todas_las_palabras_clave:
            for frase in resultado_final:
                # Verificar si la keyword está presente en la frase (ignorar mayúsculas y minúsculas)
                if keyword in frase:
                    resu_all.append((id, keyword, puntuacion, frase))

        # Imprimir los resultados
        # for resultado in resu_all:
        #     id, keyword, puntuacion, frase = resultado
        #     print(f"PLAN_ID: {id}, Keyword: {keyword}, Puntuación: {puntuacion}, Frase: {frase}")

        # %%
        # Lista para almacenar resultados
        resultados = []

        # Recorrer cada keyword de todas_las_palabras_clave
        for id, keyword, puntuacion in todas_las_palabras_clave:
            for frase in resultado_final:
                # Dividir la frase en palabras
                palabras = frase.split()
                
                # Verificar si la keyword está presente en la lista de palabras (ignorar mayúsculas y minúsculas)
                if keyword in [palabra for palabra in palabras]:
                    resultados.append((keyword, puntuacion, frase))

        # Imprimir los resultados
        # for resultado in resultados:
        #     keyword, puntuacion, frase = resultado
        #     print(f"Keyword: {keyword}, Puntuación: {puntuacion}, Frase: {frase}")

        # %%
        # Extraer los primeros elementos de cada tupla
        kws = [tupla[1] for tupla in todas_las_palabras_clave]

        # Imprimir la lista resultante
        # print(kws)

        # %%
        import re

        # Lista de cadenas originales
        cadena_original = kws

        # Lista de cadenas contenedoras
        cadena_contenedora = resultado_final

        # Usar expresión regular para encontrar la palabra completa en cada elemento de la lista
        resu_word = []

        for original in cadena_original:
            patron = r'\b\w*' + re.escape(original) + r'\w*\b'
            for contenedora in cadena_contenedora:
                resultado = re.search(patron, contenedora, re.IGNORECASE)
                if resultado:
                    palabra_completa = resultado.group()
                    resu_word.append((original, palabra_completa, contenedora))

        # Imprimir los resultados
        # for original, palabra_completa, contenedora in resu_word:
        #     print(f"Cadena original: {original}, Palabra completa: {palabra_completa}, Frase completa: {contenedora}")


        # %%
        # Lista para almacenar resultados
        active_plans = []

        # Comparar y agregar elementos a la lista active_plans
        for tupla_all, tupla_word in zip(resu_all, resu_word):
            if tupla_all[1] == tupla_word[1]:
                active_plans.append((tupla_all[0], tupla_all[1], tupla_word[2]))

        # Imprimir la lista active_plans
        # for item in active_plans:
        #     print(item)

        # %%
        active_plans = pd.DataFrame(active_plans, columns=["ID_PLAN", "KW", "TEXT"])

        # %%
        # Combinar DataFrames en función de la columna 'ID_PLAN'
        active_plans = active_plans.merge(plans_chg[['ID_PLAN', 'DS_PLAN']], on='ID_PLAN', how='left')

        # Imprimir el DataFrame resultante
        # active_plans

        # %%
        # Función para realizar la limpieza de texto
        def clean_text(text):
            # Eliminar signos de puntuación
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Convertir a minúsculas
            text = text.lower()
            
            # Tokenizar el texto en palabras
            words = text.split()
            
            # Eliminar stopwords en español
            # stop_words = set(stopwords.words('spanish'))
            words = [word for word in words if word not in stop_words]
            
            # Unir las palabras nuevamente en un solo texto
            cleaned_text = ' '.join(words)
            
            return cleaned_text

        # Aplicar la función de limpieza a la columna 'DS_PLAN'
        active_plans['DS_PLAN'] = active_plans['DS_PLAN'].apply(clean_text)

        # Imprimir el DataFrame resultante
        active_plans

        # %%
        # Filtrar las filas que no contienen "Valencia" en la columna KW
        active_plans_filtrado = active_plans.loc[~active_plans['KW'].str.contains('chicago')]

        # Imprimir el DataFrame resultante
        # active_plans_filtrado

        # %%
        # Model Selection and Initialization
        # model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 
        # model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
        # Crear una lista de diccionarios para almacenar los resultados
        fever_plans = []

        # Iterar a través de las filas del DataFrame df_sin_duplicados
        for index, row in active_plans_filtrado.iterrows():
            sentence1 = row['TEXT']
            sentence2 = row['DS_PLAN']
            
            # Encode sentences to get their embeddings
            embedding1 = model.encode(sentence1, convert_to_tensor=True)
            embedding2 = model.encode(sentence2, convert_to_tensor=True)

            # Compute similarity score of two embeddings
            cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
            similarity_score = cosine_scores.item()
            
            # Crear un diccionario con los resultados y agregarlo a la lista
            result_dict = {
                'ID_PLAN': row['ID_PLAN'],
                'KW': row['KW'],
                'TEXT': sentence1,
                'DS_PLAN': sentence2,
                'SCORE': similarity_score
            }
            fever_plans.append(result_dict)

        # Convertir la lista de diccionarios en un DataFrame
        fever_plans_df = pd.DataFrame(fever_plans)

        # Imprimir el DataFrame resultante
        # fever_plans_df

        # %%
        # Filtrar el DataFrame usando la función query
        filtered_df = fever_plans_df.query('SCORE > 0.61')

        # Imprimir el DataFrame resultante
        # filtered_df

        # %%
        # Agrupar por 'ID_PLAN' y calcular la media de los 'SCORE' en cada grupo
        mean_scores_df = filtered_df.groupby('ID_PLAN').agg({'SCORE': 'mean', 'DS_PLAN': 'first'}).reset_index()

        # Renombrar la columna 'SCORE' como 'MEAN_SCORE'
        mean_scores_df = mean_scores_df.rename(columns={'SCORE': 'MEAN_SCORE'})

        # Ordenar el DataFrame por la columna 'MEAN_SCORE' en orden descendente
        mean_scores_df = mean_scores_df.sort_values(by='MEAN_SCORE', ascending=False)

        # Reiniciar el índice del DataFrame resultante
        mean_scores_df = mean_scores_df.reset_index(drop=True)

        # Crear la columna 'PLAN_URL' usando apply y una función lambda
        mean_scores_df['PLAN_URL'] = mean_scores_df['ID_PLAN'].apply(lambda id_plan: f'https://feverup.com/m/{id_plan}')

        # Imprimir el DataFrame resultante
        st.dataframe(mean_scores_df)



        # %%
        # # Filtrar el DataFrame usando la función query
        # filtered_df = fever_plans_df.query('SCORE > 0.6')

        # # Imprimir el DataFrame resultante
        # filtered_df

        # %%
        # # Agrupar por 'ID_PLAN' y 'DS_PLAN' y seleccionar la fila con el score máximo en cada grupo
        # fever_plans_df = fever_plans_df.loc[fever_plans_df.groupby(['ID_PLAN', 'KW'])['SCORE'].idxmax()]

        # # Reiniciar el índice del DataFrame resultante
        # fever_plans_df = fever_plans_df.reset_index(drop=True)

        # # Imprimir el DataFrame resultante
        # fever_plans_df


        # %%
        # from keybert import KeyBERT

        # %%
        # kw_model = KeyBERT(model='distiluse-base-multilingual-cased-v1')

        # %%
        # from keybert import KeyBERT
        # from nltk.corpus import stopwords
        # import nltk
        # nltk.download('stopwords')

        # # Texto de entrada
        # text = "Tu texto de entrada aquí"

        # # Tokeniza el texto y elimina las stopwords en inglés
        # stop_words = set(stopwords.words("spanish"))
        # words = text.split()
        # filtered_words = [word for word in words if word.lower() not in stop_words]
        # filtered_text = ' '.join(filtered_words)

        # # Aplica KeyBERT al texto preprocesado
        # model = KeyBERT('distilbert-base-nli-mean-tokens')
        # keywords = model.extract_keywords(filtered_text)

        # print(keywords)

        # %%
        # from itertools import combinations

        # # Palabras clave después de eliminar las stopwords
        # keywords = ['texto', 'entrada', 'aquí']

        # # Rango de ventanas que deseas considerar
        # window_range = (1, 3)

        # # Número máximo de combinaciones a guardar
        # top_n = 5

        # # Lista para almacenar las combinaciones de palabras clave
        # keyword_combinations = []

        # # Itera a través de las ventanas en el rango especificado
        # for window_size in range(window_range[0], window_range[1] + 1):
        #     # Genera todas las combinaciones de palabras clave para la ventana actual
        #     combinations_for_window = list(combinations(keywords, window_size))
            
        #     # Agrega las combinaciones a la lista de resultados
        #     keyword_combinations.extend([' '.join(comb) for comb in combinations_for_window])

        # # Limita el número de combinaciones a las primeras "top_n"
        # keyword_combinations = keyword_combinations[:top_n]

        # # Imprime las combinaciones resultantes
        # for combination in keyword_combinations:
        #     print(combination)



