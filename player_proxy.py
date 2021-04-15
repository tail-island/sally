import json
import os

from subprocess import Popen, PIPE


class PlayerProxy:
    def __init__(self, program_name):
        self.name = program_name.replace('.bat', '')

        self.stderr = open(os.path.join('.', 'players', f'{program_name}-log.txt'), mode='a')
        self.process = Popen((os.path.join('.', program_name),), cwd=os.path.join('.', 'players'), shell=True, stdin=PIPE, stdout=PIPE, stderr=self.stderr, universal_newlines=True)

    def get_action(self, observation):
        self.process.stdin.write(f'{json.dumps(observation)}\n')
        self.process.stdin.flush()

        action = json.loads(self.process.stdout.readline())

        return action['acceleration'], action['braking'], action['steering']

    def done(self):
        self.process.stdin.close()
        self.process.kill()

        self.stderr.close()
