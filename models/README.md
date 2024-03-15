
# Model List
|Dataset|model name in crapcn.py| deconv based seed generator? | decoder name|
| :---: | :---: | :---: | :---: |
|PCN| CRAPCN | | Decoder|
|PCN| CRAPCN_d | ✔️| Decoder|
|ShapeNet-55/34/21 | CRAPCN_sn55 | | Decoder_sn55|
|ShapeNet-55/34/21 | CRAPCN_sn55_d | ✔️| Decoder_sn55|
|MVP| CRAPCN_mvp | | Decoder_mvp|
|MVP| CRAPCN_mvp_d |✔️ | Decoder_mvp|


# Tips
There are some issues with overflow which may be caused by the too small constant in feature interpolation,
you can try to adjust a constant in [line 81 of crapcn.py](https://github.com/EasyRy/CRA-PCN/blob/a8ec23d68e6b98fa45a4a79e423ea1c8067a1824/models/crapcn.py#L81) to alleviate them, e.g., change the constant from 1e-8 to 5e-4.
