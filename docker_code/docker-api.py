from argparse import ArgumentParser
import docker
from time import sleep
import os
import pandas as pd
from uuid import uuid4
import json

# change this to your own ID
IMAGE_NAME = 'anshuman/noyce-rl-intervention'
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
ARGS_DIR = os.path.join(os.getcwd(), 'arguments')
N = 10

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--build', action="store_true", help='Build docker image')
    parser.add_argument('--run', action="store_true", help='Run all docker containers')
    parser.add_argument('--simulate', action="store_true", help='Simulate container run')
    parser.add_argument('--max-containers', default=10, type=int, help="Maximum number of concurrent containers")
    parser.add_argument('--sleep-duration', default=60, type=int, help="Time to sleep (in seconds) when max containers are reached and before spawning additional containers")
    args = parser.parse_args()
    return args, parser

def build_image():
    # get docker client and build image
    client = docker.from_env()

    # build the image from the Dockerfile
    #   -> tag specifies the name
    #   -> rm specifies that delete intermediate images after build is completed
    client.images.build(path='.', tag=IMAGE_NAME, rm=True)

def get_mount_volumes():
    # binds "/output" on the container -> "OUTPUT_DIR" actual folder on disk
    # binds "/args" on the container -> "ARGS_DIR" actual folder on disk

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(ARGS_DIR):
        os.makedirs(ARGS_DIR)

    # mapping format for binding outputDir to /output
    return { OUTPUT_DIR: { "bind": "/output" }, ARGS_DIR: { "bind": "/args" } }

def max_containers_reached(client, max_containers):
    try:
        return len(client.containers.list()) >= max_containers
    except:
        return True

def in_range(df, key, mi, mx):
    return df[(df[key] > mi) & (df[key] < mx)]['query']


def spawn_containers(args):
    # get docker client
    client = docker.from_env()
    
    # list of labels
    LABELS = ['Far Left', 'Left', 'Moderate', 'Right', 'Far Right']

    # get video distribution
    
    # spawn containers for each user
    count = 0

    for _ in range(3000):
        #for training_label in LABELS:
        for _ in range(10):
                

                # user watch history for training
                #training = videos[training_label].sample(N).to_list()
                
                # check for running container list
                while max_containers_reached(client, args.max_containers):
                    # sleep for a minute if maxContainers are active
                    print("Max containers reached. Sleeping...")
                    sleep(args.sleep_duration)

                # spawn container if it's not a simulation
                if not args.simulate:
                    #puppetId = f'{training_label},{uuid4()}'
                    print("Spawning container...")
                    
                    # write arguments to a file
                    #with open(f'arguments/{puppetId}.json', 'w') as f:
                    #    json.dump(dict(
                    #        puppetId=puppetId,
                    #        duration=30,
                    #        description='',
                    #        outputDir='/output',
                    #        training=','.join(training),
                    #        testSeed='ST9cmttj1kY',
                    #        steps='train,test'
                    #    ), f)

                    # set outputDir as "/output"
                    command = ['python', 'tester-R.py']

                    # run the container
                    client.containers.run(IMAGE_NAME, command, volumes=get_mount_volumes(), shm_size='512M', remove=True, detach=True)
                    
                # increment count of containers
                count += 1

    print("Total containers spawned:", count)

def main():

    args, parser = parse_args()

    if args.build:
        print("Starting docker build...")
        build_image()
        print("Build complete!")

    if args.run:
        spawn_containers(args)

    if not args.build and not args.run:
        parser.print_help()


if __name__ == '__main__':
    main()
