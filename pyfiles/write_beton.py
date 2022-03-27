from ffcv.fields import RGBImageField, NDArrayField
from ffcv.writer import DatasetWriter

def write_beton_file(args):

    pytorch_dataset = create_pytorch_dataset(args)

    writer = DatasetWriter('train.beton',
                           {
                            'image': RGBImageField(),
                            'label': NDArrayField(shape=(5,), dtype=np.dtype('float32')),
                           },
                           page_size=67108864
                           )

    writer.from_indexed_dataset(pytorch_dataset)
