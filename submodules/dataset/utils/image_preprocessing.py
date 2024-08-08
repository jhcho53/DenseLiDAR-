import os
import shutil

from tqdm import tqdm

from submodules.utils.get_func import get_sync_path, get_last_5

kitti_raw_list = ['kitti_raw/2011_09_26',
                  'kitti_raw/2011_09_28',
                  'kitti_raw/2011_09_29',
                  'kitti_raw/2011_09_30',
                  'kitti_raw/2011_10_03']
img_dir = ['image_02/data/', 'image_03/data/']
first_5 = ['0000000000.png', '0000000001.png', '0000000002.png', '0000000003.png', '0000000004.png']


# 사용하지 않는 grayscale image 제거
def remove_unused_files(root_dir, kitti_raw_list):
    sync_path = get_sync_path(root_dir, kitti_raw_list)
    if os.path.exists(sync_path) and os.path.isdir(sync_path):
        try:
            shutil.rmtree(os.path.join(sync_path, 'image_00'))
            shutil.rmtree(os.path.join(sync_path, 'image_01'))
            shutil.rmtree(os.path.join(sync_path, 'velodyne_points'))
        except Exception as e:
            print(e)

def remove_oxts(root_dir, kitti_raw_list):
    sync_path = get_sync_path(root_dir, kitti_raw_list)
    if os.path.exists(sync_path) and os.path.isdir(sync_path):
        try:
            shutil.rmtree(os.path.join(sync_path, 'oxts'))
        except Exception as e:
            print(e)

def move_folder(root_dir, kitti, list, dst):
    # root_dir | image or lidar folder name | train or val inner folder list ! destination directory
    for folder in list:
        source_folder_path = os.path.join(root_dir, kitti, folder)
        if os.path.isdir(source_folder_path):
            shutil.move(source_folder_path, dst)

def split_train_val(root_dir, kitti_raw_list, train_list, image_train_dir, val_list, image_val_dir):
    for kitti_raw in tqdm(kitti_raw_list):
        # train_list에 해당하는 폴더를 train 디렉토리로 이동
        move_folder(root_dir, kitti_raw, train_list, image_train_dir)

        # val_list에 해당하는 폴더를 val 디렉토리로 이동
        move_folder(root_dir, kitti_raw, val_list, image_val_dir)
        shutil.rmtree(os.path.join(root_dir, kitti_raw))

    shutil.rmtree(os.path.join(image_train_dir, 'image_02'))
    shutil.rmtree(os.path.join(image_train_dir, 'image_03'))
    shutil.rmtree(os.path.join(image_val_dir, 'image_02'))
    shutil.rmtree(os.path.join(image_val_dir, 'image_03'))

# lidar data와 장면을 맞추기 위해 이미지의 처음/마지막 5개 파일 제거.
def remove_first_last_5(root_dir, image_train_dir, image_val_dir):
    train_dir = os.path.join(root_dir, 'kitti_raw/train')
    val_dir = os.path.join(root_dir, 'kitti_raw/val')

    train_list = os.listdir(train_dir)
    val_list = os.listdir(val_dir)
    mode_list = [train_list, val_list]

    for mode in tqdm(mode_list):
        for dir in mode:
            if mode == train_list:
                current_dir = os.path.join(root_dir, os.path.join(image_train_dir, dir))
            elif mode == val_list:
                current_dir = os.path.join(root_dir, os.path.join(image_val_dir, dir))
            for img in img_dir:
                try:
                    dir_path = os.path.join(current_dir, img)
                    files = os.listdir(dir_path)

                    # 파일이 있는 경우에만 처리
                    if files:
                        # 첫 5개 파일 삭제
                        for file in first_5:
                            file_path = os.path.join(dir_path, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            except Exception as e:
                                print(e)
                        last_5 = get_last_5(dir_path)
                        for file in last_5:
                            file_path = os.path.join(dir_path, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            except Exception as e:
                                print(e)

                    else:
                        print("Directory is Empty.")
                except Exception as e:
                    print(e)
                    pass


def image_preprocessing(root_dir):
    train_dir = os.path.join(root_dir, 'data_depth_annotated/train')
    val_dir = os.path.join(root_dir, 'data_depth_annotated/val')

    # train/test split에 사용할 sync 폴더 리스트.
    train_list = os.listdir(train_dir)
    val_list = os.listdir(val_dir)

    image_train_dir = os.path.join(root_dir, 'kitti_raw/train')
    image_val_dir = os.path.join(root_dir, 'kitti_raw/val')

    print("\nRemove unused files:")
    remove_unused_files(root_dir, kitti_raw_list)
    remove_oxts(root_dir, kitti_raw_list)

    print("\nSplit train and val:")
    split_train_val(root_dir, kitti_raw_list, train_list, image_train_dir, val_list, image_val_dir)

    print("\nRemove first last 5:")
    remove_first_last_5(root_dir, image_train_dir, image_val_dir)

    image_train_list = os.listdir(image_train_dir)
    image_val_list = os.listdir(image_val_dir)

    return image_train_list, image_val_list