import os
import h5py
import numpy as np
import torch

def generate_compiler(data_root, hdf_data_path,BZ,IR,HARDSTOP,
                        label_scheme='label' ):
    """
    Gathers all files and organize into train/validation/test. Each
       data subset is further organized by class. Converts file lists
       into iterable data generators.
    Inputs:
        * data_root: full path to the data root containing subdirectories of data
        * trn_list, val_list, tst_list: each is a list of strings. The strings
            should encode the strata being used to divide data into segments and
            must match the corresponding field of the FileName class.
        * hdf_data_path: the path used internally in the HDF to get the desired surface
        * batch_size: the _approximate_ number of samples to be used __next__ call. Note
              that if batch_size // n_classes is non-integer then it may be slightly off.
        * label_scheme: a string that matches the name of the property of the FileName
              class that you want to use as the label.
        * strata_scheme: a string that matches the name of the property of the FileName
              class that you want to compare against trn_list (etc.) for data segmentation.
    """
    n_classes = 2
    files = [[] for clas in range(n_classes)]
    
    #print('Allocating HDFs to train/valid/test...')
    
    cnt = 0
    for filename in os.listdir(data_root):
        if not filename.endswith('.hdf'):
            continue

        # Extract the label using 'label_scheme' identifier
        if len(files[0]) >= HARDSTOP and len(files[1]) >= HARDSTOP:
            break
        label = int(int(filename.split('_')[3]) > 0)  # 0 = clutter (not manmade); 1 = target (manmade)
            
        files[label].append(filename)

        cnt += 1

        

    print(f"Clutter len: {len(files[0])}")
    print(f"Target len: {len(files[1])}")       

    # Wrap each file list into an iterable data generator that actually
    #     read the HDFs when __next__ is called:
    
    gen = DataGenerator(data_root, files, hdf_data_path,  BZ, IR)
    

    return gen


class DataGenerator:
    def __init__(self, data_root, file_list, hdf_data_path, BZ, IR, n_classes=2):
        # Basic properties:
        self.data_root = data_root
        self.n_classes = n_classes

        "balance by class"
        self.hdf_path = hdf_data_path

        self._permanent_file_list = file_list
        # print(len(file_list[0]), "\n")
        "for august-november split"
        # self.target_list = file_list[1][0:4
        "for fulll dataset"
        self.target_list = file_list[1]
        self.target_len = len(self.target_list)
        # print(self.target_len, 'length of target')
        
        "test"

        #self.clutter_list = file_list[0]
        self.clutter_list = file_list[0][0:int(IR*self.target_len)]

        #self.clutter_list = file_list[0][0:16799]
        self.clutter_len = len(self.clutter_list)
        # print(self.clutter_len, 'length of clutter')

        self.dataset_size = self.clutter_len + self.target_len
        self.batch_size = BZ
        self.bsz_by_class = 0
        self.iters = 0
        # if self.batch_size % 2 != 0 or self.bsz_by_class % 2 != 0:
        #     print('batch size is odd or not balanced... \nadding one to batch size')
        #     self.batch_size = self.batch_size  + 1
        #     self.bsz_by_class = math.floor(self.batch_size / self.n_classes)

        



        # print(self.batch_size, 'batch_size')
        # print(self.bsz_by_class, 'balance size')

        #self.current_index = [i for i in range(self.batch_size)]


        #self.current_index = [0 for label in range(self.n_classes)]

        self.HDF_n_rows = 71
        self.HDF_n_cols = 71
        self.HDF_n_dpth = 101

        self.input_shape = (self.batch_size, self.HDF_n_rows, self.HDF_n_cols, self.HDF_n_dpth)

        self.chip_n_rows = 64
        self.chip_n_cols = 64
        self.chip_n_dpth = 101

        "chip shape is one cube from batch"
        self.chip_shape = (self.chip_n_rows, self.chip_n_cols, self.chip_n_dpth)

        self.batch_shape = (self.batch_size, self.chip_n_rows, self.chip_n_cols, self.chip_n_dpth)


        # Make a boolean indexing mask for efficient extraction of the chip center:
        "input shape minus chip shape row"
        row_diff = self.input_shape[1] - self.chip_shape[0]
        "this is chopping off one half the difference on each side"
        first_row = row_diff // 2
        last_row = first_row + self.chip_shape[0]

        "this is chopping off one half the difference on each side"
        col_diff = self.input_shape[2] - self.chip_shape[1]
        first_col = col_diff // 2
        last_col = first_col + self.chip_shape[1]

        "this is chopping off one half the difference on each side"
        slice_diff = self.input_shape[3] - self.chip_shape[2]
        first_slice = slice_diff // 2
        last_slice = first_slice + self.chip_shape[2]

        "slicing out the chips from the input data"
        "list with shape of input all False values"
        self.center_select = np.full((self.HDF_n_rows, self.HDF_n_cols, self.HDF_n_dpth), False)
        "except the chips size all true"
        self.center_select[first_row:last_row, first_col:last_col, first_slice:last_slice] = True

        # self.first_slice = 0
        # self.last_slice = self.bsz_by_class


    def readHDF(self, file_name):
        """ Reads data from the HDF"""
        with h5py.File(os.path.join(self.data_root, file_name), mode='r') as f:
            data = f[self.hdf_path][:]
            data = np.transpose(data)  # because python reverses the order of the 3d volume, this will correct back to the original matlab order
            #data = data / 40  # Now data is in [0,1], to match the 2d sensors


        return data

    def chip_center(self, data):
        """ extracts the center of the chip via boolean indexing. """
        return np.reshape(data[self.center_select], self.chip_shape)


    def preprocess(self, data_sample):
        """
        Preprocesses a data sample.
        TODO: use configuration file and preprocesser class like ADAM dataloader here.
        """
        # TODO: sometimes want to jiggle the chip instead of centering so will
        #          need to remove chip_center from here.

        "centering x cube"
        x_center = self.chip_center(data_sample)
        return x_center

    

    def data_loop(self, list_data):

        "place holder for batch of data"
        batch_data = np.zeros(self.batch_shape, dtype='float32')
        "batch labels"
        batch_label = np.zeros(self.batch_size)

        for label in range(0, self.n_classes):
            for nth_sample in range(0, self.bsz_by_class):

                ld = list_data[label]
                sample = ld[nth_sample]

                data = self.readHDF(sample)

                # Preprocessing can go here, e.g. random flips/translations/normalizations
                "centers the data here"
                data = self.preprocess(data)

                # Insert the data into the batch_data and batch_label arrays:
                batch_idx = self.bsz_by_class * label + nth_sample
                "batch data: why just rows and columns"
                batch_data[batch_idx][:, :, :] = np.reshape(data, self.chip_shape)
                "batch label"
                batch_label[batch_idx] = label

        return batch_data, batch_label

    def __iter__(self):
        self.first_slice = 0
        self.last_slice = self.bsz_by_class
        return self

    def __len__(self):
        return sum([len(label_set) for label_set in self.target_list])

    def __next__(self):


        # Put `bsz_by_class' samples from each class into `batch_data':

        # for n in range(self.bsz_by_class, self.target_len, self.bsz_by_class):


        if self.iters < self.dataset_size:
            #print(self.last_slice, 'last slice')

            if self.iters % 2 == 0:
                data = self.readHDF(self.clutter_list[self.iters//2])

                data = self.preprocess(data)


                batch_data, batch_label = data, torch.zeros(1)

            else:
                data = self.readHDF(self.target_list[self.iters//2])

                data = self.preprocess(data)


                batch_data, batch_label = data, torch.ones(1)

        else:
            print('\nNEXT\n')
            raise StopIteration

        self.iters += 1
        return batch_data, batch_label
        # , self.bsz_by_class, self.clutter_len, self.target_len, self.batch_size


