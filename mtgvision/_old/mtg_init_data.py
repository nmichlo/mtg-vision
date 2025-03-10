#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2025 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


from mtgvision.datasets import MtgLocalFiles, IlsvrcImages

# from mtg_detect import CardMatcher
# from util import Proxy

if __name__ == "__main__":
    # RANDOM BACKGROUNDS

    IlsvrcImages()

    # SOURCE IMAGES

    # MtgHandler(force_update_scryfall=True)

    # used below
    # MtgImages(img_type='small', predownload=True)

    # MTG DETECTOR

    # with tf.Session() as sess:
    #     matcher = CardMatcher('weights.03-0.5411.hdf5')
    #     matcher.initialise()

    # LOCAL FILES

    dataset = MtgLocalFiles(img_type="small", x_size=(192, 128), y_size=(96, 64))
    chosen = [
        "prm/prm__69eadfec-7799-458e-afc8-4237c8dd95f4__forest__small.jpg",
        "tust/tust__80165be4-c6c8-4b22-b259-c64eb4b7fc95__storm-crow__small.jpg",
        "hml/hml__9b080587-d062-42ff-abc5-8e04a20faece__clockwork-steed__small.jpg",
        "leg/leg__7f841918-813b-4784-ab57-907185b0a355__jerrard-of-the-closed-fist__small.jpg",
        "wc01/wc01__a4369a59-5a39-40f7-b97b-763b575203aa__island__small.jpg",
        "dom/dom__7f3423d7-cb81-47bf-b9a6-a279ba6cedf4__phyrexian-scriptures__small.jpg",
        "dom/dom__ba05cf47-9823-41f9-b893-321ea89e473e__woodland-cemetery__small.jpg",
        "wc04/wc04__18957d46-01e9-4ef0-97ba-ddbe5b34ea10__myr-incubator__small.jpg",
        "ohop/ohop__6e17f67b-60a0-4858-9454-92b35dee08a6__naya__small.jpg",
    ]
    out_x, out_y, out_o = dataset.gen_warp_crop_orig_set(20, save=False, chosen=chosen)
