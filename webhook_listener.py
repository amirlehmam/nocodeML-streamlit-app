from flask import Flask, request

import os

import logging

import threading


logging.basicConfig(filename='/home/ec2-user/NoCodeML/webhook_listener.log', level=logging.INFO)


app = Flask(__name__)


def run_update_script():

    os.system('/home/ec2-user/NoCodeML/update_and_restart.sh')


@app.route('/payload', methods=['POST'])

def payload():

    if request.method == 'POST':

        logging.info('Received webhook payload')

        threading.Thread(target=run_update_script).start()

        return 'Success', 200

    else:

        logging.info('Invalid request')

        return 'Invalid Request', 400


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)

