# Researches on Covariates

## Analysis on PaddleTS

Relevent tutorial is [here](https://paddlets.readthedocs.io/en/0.1.1/source/get_started/get_started.html#id7).

Baseline code is following
```python
from paddlets import TSDataset
from matplotlib import pyplot as plt
from paddlets.models import forecasting
import paddle


def load_data(csv_path: str):
    return TSDataset.load_from_csv(
        filepath_or_buffer=csv_path,
        # Custom your own fields here
        observed_cov_cols=['fj_activePower', 'fj_windDirection', 'cft_ws10', 'cft_wd10', 'cft_ws50', 'cft_wd50',
        'cft_wsHubHeight', 'cft_t8', 'cft_p8'],
        target_cols=['fj_windSpeed'],
        time_col='date'
    )


def main():
    paddle.set_device('gpu')
    dataset = load_data(
        # Replace here to your path of data
        csv_path='./FJ1-clean-interpolate.csv'
    )
    dataset.plot()
    plt.show()
    model = forecasting.TFTModel(
        in_chunk_len=99,
        out_chunk_len=23,
        max_epochs=1000
    )
    model.fit(dataset)
    model.save('model.pt')


if __name__ == '__main__':
    main()
```

With the source code modified and debug, we learnt about how it works.

## Preprocessing Data

In class method ```TSDataset.load_from_csv```, it reads the content of CSV file to a ```pandas.DataFrame```. [Source code here](https://github.com/PaddlePaddle/PaddleTS/blob/fa1f6b25a857c4ca611f536894ca7bd020824bec/paddlets/datasets/tsdataset.py#L810).

Firstly, fetch every one of fields from field name passed to arguments and store them to their respective ```Tensor```.

> Except for ```static_cov```, which is [strictly examined](https://github.com/PaddlePaddle/PaddleTS/blob/fa1f6b25a857c4ca611f536894ca7bd020824bec/paddlets/datasets/tsdataset.py#L952), and then store every fields to dictionary.

Secondly, they're processed using [certain time series transformation](https://github.com/PaddlePaddle/PaddleTS/blob/fa1f6b25a857c4ca611f536894ca7bd020824bec/paddlets/datasets/tsdataset.py#L907). The kernel function to perform this can be found [here](https://github.com/PaddlePaddle/PaddleTS/blob/fa1f6b25a857c4ca611f536894ca7bd020824bec/paddlets/datasets/tsdataset.py#L77). As it's shown, it infers the frequency type ```freq``` from the date. In the class ```TimeSeries```, it invokes [API](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.asfreq.html) from ```pandas``` to embed time series information to the data.

Thirdly, they are made into several batches.

> That is, we divide them into several ```batch_size*in_chuck_len```-length sequences, and reshape them.

Finally, it performs pre-training processes. It makes every one of fields into several sample dictionary, as seen [here](adapter).

## Training and Prediction Time

The ```Transformer``` model can be found [here](https://github.com/PaddlePaddle/PaddleTS/blob/fa1f6b25a857c4ca611f536894ca7bd020824bec/paddlets/models/forecasting/dl/transformer.py#L205).

> Transformer model does not actually use ```static_cov```, but it could have different usage in different models.

It actually ```concat``` the ```target```, ```observed_cov```*if any*, and ```known_cov```*if any* as input, using function ```concat(axis=-1)```

The shapes  are as follows:

| Fields | Shape |
| :-: | :-: |
| src | (batch_size, in_chuck_len, target_dim) |
| tgt | (batch_size, 1, target_dim) |
| output | (batch_size, output_chuck_len, n_target_cols) |

In which:

| Symbols | Values |
| :-: | :-: |
| target_dim | len(target_cols) + len(known_cov_cols) |
| n_target_cols | len(target_cols) |

This Transformer is in Encoder-Decoder mode, except for absence of mask. The src and tgt are inputs to Encoder and Decoder respectively, and finally we obtain output representing the predicted targets.
