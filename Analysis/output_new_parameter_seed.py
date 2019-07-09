#!/usr/bin/env python
import configparser
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',help='new parameter file')
    parser.add_argument('new_seed',type=int)
    parser.add_argument('output_file',help='new parameter file')
    args = parser.parse_args()

    myconfigparser = configparser.ConfigParser()
    myconfigparser.read(args.input_file)
    params = myconfigparser['params']

    params['seed'] = str(args.new_seed)

    print(args.output_file)

    with open(args.output_file,'w') as f:
        myconfigparser.write(f)
