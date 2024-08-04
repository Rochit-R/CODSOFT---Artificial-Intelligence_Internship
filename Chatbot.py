import re

def chatbot_response(user_input):
    user_input = user_input.lower().strip()

    patterns = {
        'math': r'\b(math|mathematics|algebra|geometry|calculus)\b',
        'science': r'\b(science|physics|chemistry|biology|earth science)\b',
        'history': r'\b(history|ancient history|modern history|world history)\b',
        'english': r'\b(english|literature|grammar|writing)\b',
        'geography': r'\b(geography|maps|climates|physical geography|human geography)\b',
        'tamil': r'\b(tamil|தமிழ்)\b',
        'hindi': r'\b(hindi|हिन्दी)\b',
        'french': r'\b(french|français)\b',
    }

    for subject, pattern in patterns.items():

        if re.search(pattern, user_input):

            if subject == 'math':
                return "Math is a fascinating subject! Are you looking for help with algebra, geometry, or calculus?"
            elif subject == 'science':
                return "Science covers many areas like physics, chemistry, and biology. Which one are you interested in?"
            elif subject == 'history':
                return "History helps us understand past events. Are you studying a specific period or event?"
            elif subject == 'english':
                return "English involves literature, grammar, and writing. What do you need help with?"
            elif subject == 'geography':
                return "Geography studies the Earth's landscapes and environments. Do you have questions about maps or climates?"
            elif subject == 'tamil':
                return "Tamil is a beautiful language with a rich literary heritage. Are you looking for help with Tamil language or literature?"
            elif subject == 'hindi':
                return "Hindi is one of the major languages of India. Are you interested in learning more about Hindi language or literature?"
            elif subject == 'french':
                return "French is a widely spoken language with a rich culture. Do you need help with French language or culture?"

    return "I'm not sure about that subject. Can you please specify a school subject or language like math, science, history, English, Tamil, Hindi, or French?"

def main():

    print("Welcome to the advanced school subjects and languages chatbot! Type 'bye' to end the conversation.")
    
    while True:

        user_input = input("You: ")

        if 'bye' in user_input.lower():
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()