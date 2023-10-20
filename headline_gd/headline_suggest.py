# %%
import openai
import requests
from bs4 import BeautifulSoup
import streamlit as st

# url = "https://valenciasecreta.com/cervezas-artesanas-valencianas/"


# INtro prompt


# %%
openai.api_key = "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"

def headline_suggestions(prompt, temp, model="text-davinci-003"): #text-davinci-003 | gpt-3.5-turbo-instruct
  response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    temperature=temp,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  headline = response.choices[0].text.strip()
  headline = headline.replace('1.', '1. ')
  headline = headline.replace('2.', '\n2. ')
  headline = headline.replace('3.', '\n3. ')
  headline = headline.replace('4.', '\n4. ')
  headline = headline.replace('5.', '\n5. ')
  headline = headline.replace('6.', '\n6. ')
  headline = headline.replace('7.', '\n7. ')
  headline = headline.replace('8.', '\n8. ')
  headline = headline.replace('9.', '\n9. ')
  headline = headline.replace('10.', '\n10. ')
  return headline
  

# %%
#Intro prompt
def main():
    url = st.text_input("Insert the URL you want headline suggestions for: ")

    if url:
        get_url = requests.get(url)
        get_text = get_url.text
        soup = BeautifulSoup(get_text, "html.parser")
        article_section = soup.find("section", class_="article__body col-md-8")

        # Encontrar el primer h2 dentro de la sección
        first_h2 = article_section.find('h2')

        if first_h2:

            # Encontrar todos los elementos p dentro de la sección
            p_elements = article_section.find_all('p')

            # Filtrar los elementos p que están antes del primer h2
            title_content = []
            for p in p_elements:
                if p.find_previous('h2') == first_h2:
                    break
                title_content.append(p.get_text())

            # Imprimir los contenidos de los elementos p
            # for p in title_content:
                # print(p)  # Obtener el contenido de texto del elemento p

            title_content = " ".join(title_content)
        else:
            title_content = []
            for p in article_section:
                title_content.append(p.get_text())
            title_content = " ".join(title_content)

            # Imprimir los contenidos de los elementos p
            # for p in title_content:
                # print(p)  # Obtener el contenido de texto del elemento p            

        user_prompt = "You are multilingual Title Generator GPT, a professional content marketer who helps writers, bloggers, and content creators with crafting captivating titles for their articles. You are a world-class expert in generating different types of article titles that grab the readers attention and entice them to read the article.To maximize the performance and the chance of appearing in Google Discover, articles should follow some best practices. These Google Discover best practices are the following:\n\n1.Geographical References: Mention specific cities or regions, as content related to local or regional events tends to perform well. 2.Sense of Urgency or Immediacy: If the information is available in the paragraph provided, use words like 'today', 'begins', or other time-sensitive terms to create a sense of urgency, encouraging immediate action or engagement. 3. Intriguing or Unique Angle: Apply an element of uniqueness or intrigue in the headlines, suggesting an unusual perspective or an alternative view of a popular location. 4. Event Announcements: If it fits the provided paragraph, headlines will announce upcoming events, festivals, or fairs, which tend to capture readers' interest, especially if they are unique or significant. 5. Promises of Unique Experiences: Headlines will appeal to people's desire for distinctive and memorable experiences. 6. Use of Superlatives: Words like 'best' or 'most beautiful' are used in headlines, indicating a sense of exclusivity or exceptional quality, which can intrigue readers. 7. Local Events and Attractions: If information is available in the provided paragraph, headlines will highlight local attractions or events, catering to the interests of the local audience. 8. Specific Event or Activity Emphasis: If information is available in the provided paragraph, headlines will create a sense of urgency or exclusivity by highlighting events, openings, or closures. This strategy aims to evoke a strong emotional response and prompt readers to take immediate action. 9. Mystery or Curiosity Creation: headlines will pose questions or introduce intriguing elements, encouraging readers to delve deeper into the content to uncover the answers. This approach aims to stimulate curiosity and increase engagement. 10. Promotion of Local and Unique Experiences:  If it fits the provided paragraph, headlines will focus on promoting local attractions, eateries, or events, often highlighting unique or unusual aspects to attract a specific target audience seeking distinctive experiences. 11. Seasonal or Time-Sensitive Content: Some headlines are designed to be timely, focusing on events, changes, or developments occurring during specific seasons or around particular dates. This tactic aims to create a sense of urgency and prompt immediate interest from the audience.\n\nThis is your TASK: Following the Google Discover patterns and commonalities you've observed above, write ten different headlines that are between based on the paragraph bellow to maximize the performance and the chances of appearing in Google Discover. Each title should be a unique type. Remember that these headlines will share the common goal of capturing attention, generating interest, and encouraging readers to engage with the content by emphasizing unique, timely, or exclusive aspects of the information presented. The headlines must follow as many title criteria described above as possible given the information in the paragraph. The headlines cannot be shorter than 65 characters long and no longer than 70 characters long, including spaces and punctuation, and must be in the same language as the paragraph. If the paragraph is in Spanish, the headlines must be in Spanish too.\n\nThis is the paragraph: "+title_content+". "

        st.subheader("Randomness")
        st.text("As the randomness approaches zero, the headline suggestions will become\ndeterministic and repetitive")
        temp = st.slider('Select a value', 0.00, 1.00, 0.65)
        chatbot_response = headline_suggestions(user_prompt, temp)
        st.subheader("Suggested headlines:")
        st.info(chatbot_response)
if __name__ == "__main__":
    main()