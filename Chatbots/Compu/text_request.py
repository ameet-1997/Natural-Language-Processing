import os.path
import sys
import json

try:
    import apiai
except ImportError:
    sys.path.append(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    )
    import apiai

CLIENT_ACCESS_TOKEN = 'client_access_token'


def main():
    ai = apiai.ApiAI(CLIENT_ACCESS_TOKEN)

    request = ai.text_request()

    request.lang = 'de'  # optional, default value equal 'en'

    # request.session_id = "<SESSION ID, UNIQUE FOR EACH USER>"

    question = raw_input()
    request.query = question
    # request.query = "Hello"

    response = request.getresponse()
    res = response.read()
    dict_response = json.loads(res)
    print(dict_response[unicode("result")][unicode("fulfillment")][unicode("speech")])


if __name__ == '__main__':
    main()
