import string
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='simple CNN model')

parser.add_argument('-r', '--root', type=str, default='./data',
                    help='root of dataset. default to ./data')

parser.add_argument('-c', '--channels', type=int,
                    choices=[3, 12],
                    help='number of channels.')

parser.add_argument('-d', '--sugiura', type=str,
                    choices=['book', 'official'],
                    help='dataset name')

args = parser.parse_args()
print(args)

if args.channels == 3:
    print(args.sugiura)
    print("channel is 3")
else:
    print(args.sugiura)
    print('channel is 12')

# img = np.random.rand(3, 200, 300)
# multi_img = np.zeros((12, 200, 300))

# color_pallete = np.random.rand(3, 12)


# # for i in range(200):
# #     for j in range(300):
# #         dist = []
# #         for k in range(12):
# #             dist.append(np.linalg.norm(img[i, j], color_pallete[:, k]))
# #         multi_img[np.argmin(dist), i, j] = 1.


# for i in range(200):
#     for j in range(300):
#         dist = np.linalg.norm(color_pallete - img[i, j])

#         multi_img[np.argmin(dist), i, j] = 1.


# # dist = np.linalg.norm(color_pallete - img)
# # multi_img[np.argmin(dist, axis=), i, j] = 1.
