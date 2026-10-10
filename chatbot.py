import os
import json

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")


# =========================
# API CLIENTS
# =========================

groq = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)


MODEL = "openai/gpt-oss-120b"


# =========================
# FAKE RESTAURANT DATA
# =========================

RESTAURANT_INFO = """
Restaurant Name: Tasty Bites

Menu:

Burgers:
- Classic Burger: Rs. 450
- Zinger Burger: Rs. 550
- Double Cheese Burger: Rs. 700

Pizza:
- Small Pizza: Rs. 800
- Medium Pizza: Rs. 1200
- Large Pizza: Rs. 1800

Broast:
- 2 Piece Broast: Rs. 450
- 4 Piece Broast: Rs. 800

Drinks:
- Pepsi: Rs. 100
- Coke: Rs. 100
- Mineral Water: Rs. 80

Deals:
- Deal 1: Zinger Burger + Fries + Drink = Rs. 650
- Deal 2: 2 Zinger Burgers + 2 Drinks = Rs. 1100
- Deal 3: Medium Pizza + 2 Drinks = Rs. 1350

Delivery:
- Delivery time: 30-45 minutes
- Delivery charges: Rs. 100

Opening Hours:
- 11:00 AM to 12:00 AM
"""


# =========================
# WEB SEARCH
# =========================

def web_search(query):

    try:

        print("\n🔎 Searching the web...\n")

        result = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        results = []

        for item in result.get("results", []):

            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content")
            })

        return json.dumps(
            results,
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps({
            "error": str(e)
        })


# =========================
# TOOL
# =========================

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for restaurant-related "
                "information when necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Restaurant-related search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = f"""
You are a sales chatbot for a fictional practice restaurant.

Restaurant information:

{RESTAURANT_INFO}


YOUR MAIN JOB:

You are a RESTAURANT SALES ASSISTANT.

Your job is to help customers choose food and place orders.


ALLOWED TOPICS:

You can answer questions about:

- Restaurant menu
- Food
- Burgers
- Pizza
- Broast
- Drinks
- Prices
- Deals
- Discounts
- Delivery
- Delivery charges
- Delivery time
- Opening hours
- Ordering food
- Recommendations
- Helping customers choose a meal


SALES BEHAVIOR:

Be friendly and helpful.

Try to help the customer decide what to order.

For example:

Customer:
"I want something cheap."

You should recommend an affordable item or deal.

Customer:
"What do you recommend?"

Recommend popular food or a good-value deal.

Customer:
"What is your best deal?"

Explain the available deals.

When appropriate, ask:
"Would you like to place an order?"


VERY IMPORTANT:

You MUST NOT answer questions unrelated to the restaurant.

Examples of questions you MUST refuse:

- What is Python?
- Write JavaScript code.
- What is AI?
- Who is the president?
- Tell me a joke.
- Solve my math homework.
- What is the weather?
- Help me hack a computer.
- Tell me about science.


For unrelated questions, reply ONLY with something similar to:

"I'm the Tasty Bites restaurant sales assistant. I can only help with our menu, prices, deals, delivery, and orders."


WEB SEARCH:

You have access to a web search tool.

Use web search ONLY when the user asks for restaurant-related information that is not available in the provided restaurant information.

Never use web search for unrelated questions.

Never use web search to answer general knowledge questions.


IMPORTANT:

Do not invent prices or menu items that are not provided.

If you don't know something about the restaurant, say that you don't have that information.
"""


# =========================
# CHAT FUNCTION
# =========================

def chat(user_message, conversation):

    conversation.append({
        "role": "user",
        "content": user_message
    })

    while True:

        response = groq.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *conversation
            ],

            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message


        # =========================
        # NORMAL AI RESPONSE
        # =========================

        if not message.tool_calls:

            answer = message.content or ""

            conversation.append({
                "role": "assistant",
                "content": answer
            })

            return answer


        # =========================
        # TOOL CALL
        # =========================

        conversation.append(message)

        for tool_call in message.tool_calls:

            if tool_call.function.name == "web_search":

                arguments = json.loads(
                    tool_call.function.arguments
                )

                query = arguments["query"]

                search_result = web_search(query)

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": search_result
                })


# =========================
# MAIN PROGRAM
# =========================

def main():

    print("=" * 55)
    print("🍔 TASTY BITES - AI SALES CHATBOT")
    print("=" * 55)

    print("Groq: Connected ✅")
    print("Tavily: Connected ✅")

    print("\nThis is a practice restaurant chatbot.")
    print("Type 'exit' to stop.\n")


    conversation = []


    while True:

        try:

            user_message = input("You: ").strip()


            if user_message.lower() == "exit":

                print("\nThank you! Goodbye 👋")
                break


            if not user_message:
                continue


            answer = chat(
                user_message,
                conversation
            )


            print("\n🤖 Tasty Bites:", answer)
            print()


        except KeyboardInterrupt:

            print("\n\nGoodbye! 👋")
            break


        except Exception as e:

            print("\n❌ Error:", e)
            print()


# =========================
# START
# =========================

if __name__ == "__main__":
    main()