import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rufus import Rufus

def main():
    prompt = "I want to build a website on services provided by chima"
    url = "https://www.withchima.com/"

    rufus_agent = Rufus(prompt, url)
    summary = rufus_agent.run()

    print("Final Summary:")
    print(summary)

if __name__ == '__main__':
    main()
