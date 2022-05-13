import argparse
import myPix2pix
import CMPFacadeBookPix2pix
import bookOriginalPix2pix
import MulChannelCMPFacadeBookPix2pix

parser = argparse.ArgumentParser(description='simple CNN model')

parser.add_argument('-c', '--channels', type=int,
                    choices=[3, 12],
                    help='number of channels.')

parser.add_argument('-d', '--dataset', type=str,
                    choices=['book', 'official'],
                    help='dataset name')

args = parser.parse_args()
if args.channels == 3:
    if args.dataset == 'book':
        bookOriginalPix2pix.main()
    else:
        CMPFacadeBookPix2pix.main()

else:
    if args.dataset == 'official':
        MulChannelCMPFacadeBookPix2pix.main()
    else:
        myPix2pix.main()
