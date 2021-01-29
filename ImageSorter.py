import cv2
import os
import progressbar
from shutil import copyfile

known_loc = 'sample_known/'
to_sort_folder = 'sample_to_sort/'
new_folder_loc = 'sorted/'


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def pre_process_image(image_loc):
    image = cv2.imread(image_loc)
    height, width, channels = image.shape
    image_cropped = image[0:int(2 * (height / 1)), 0:width]  # this line crops
    image_bw = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    return image_bw


class KnownImage:
    def __init__(self, image_name, kp, des, loc):
        self.name = image_name
        self.kp = kp
        self.des = des
        self.dist = 0.70

        # Brute-force matching
        self.bf = cv2.BFMatcher()

        # FLANN-Based matching
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        #Location
        self.loc = loc

    # Return the count of matches using Brute-Force Matching
    def run_bf_matching(self, des2):
        bf_matches = self.bf.knnMatch(self.des, des2, k=2)
        matches = 0
        if len(bf_matches) > 0 and len(bf_matches[0]) > 1:
            for m, n in bf_matches:
                if m.distance < self.dist * n.distance:
                    matches += 1
        return matches

    # Return the count of matches using FLANN-Base Matching
    def run_flann_matching(self, des2):
        flann_matches = self.flann.knnMatch(self.des, des2, k=2)
        matches = 0
        if len(flann_matches) > 0 and len(flann_matches[0]) > 1:
            for i, (m, n) in enumerate(flann_matches):
                if m.distance < self.dist * n.distance:
                    matches += 1
        return matches


class SiftSorter:
    def __init__(self):
        print('Initializing known images...')
        self.sift = cv2.SIFT_create()
        self.unk_limit = 2
        self.known_images = []
        self.unknown_loc = f'{new_folder_loc}unknown/'
        make_folder(self.unknown_loc)
        for known_name in sorted(os.listdir(known_loc)):
            if known_name != '.DS_Store':
                known_full_loc = f'{known_loc}{known_name}'
                known_name = os.path.splitext(known_name)[0]
                self.add_known_image(known_full_loc, known_name)

    def add_known_image(self, image_loc, known_name):
        base_img_bw = pre_process_image(image_loc)
        kp, des = self.sift.detectAndCompute(base_img_bw, None)
        loc = f'{new_folder_loc}{known_name}/'
        make_folder(loc)
        known_image = KnownImage(known_name, kp, des, loc)
        self.known_images.append(known_image)

    def sort_image(self, test_folder):
        num_files = len(os.listdir(test_folder))
        bar = progressbar.ProgressBar(maxval=num_files,
                                      widgets=[progressbar.Bar('=', '[', ']'), '', progressbar.Percentage()])
        bar.start()
        i = 0
        for image_file in os.listdir(test_folder):
            bar.update(i+1)
            if image_file.endswith(".jpeg") or image_file.endswith(".jpg"):
                full_test_loc = f'{test_folder}{image_file}'
                query_img_bw = pre_process_image(full_test_loc)
                kp, des = self.sift.detectAndCompute(query_img_bw, None)

                bf_loc = self.unknown_loc
                bf_max_matches = self.unk_limit
                flann_loc = self.unknown_loc
                flann_max_matches = self.unk_limit
                for candidate in self.known_images:
                    bf_matches = candidate.run_bf_matching(des)
                    if bf_max_matches <= bf_matches:
                        bf_max_matches = bf_matches
                        bf_loc = candidate.loc

                    flann_matches = candidate.run_flann_matching(des)
                    if flann_max_matches <= flann_matches:
                        flann_max_matches = flann_matches
                        flann_loc = candidate.loc

                if bf_loc != self.unknown_loc:
                    # print(f'Copying {full_test_loc} to {bf_loc}')
                    copyfile(full_test_loc, f'{bf_loc}{image_file}')
                else:
                    # print(f'Copying {full_test_loc} to {flann_loc}')
                    copyfile(full_test_loc, f'{flann_loc}{image_file}')
            i += 1
        bar.finish()


if __name__ == "__main__":
    sift_sorter = SiftSorter()
    print(f'Sorting: {to_sort_folder}')
    sift_sorter.sort_image(to_sort_folder)
