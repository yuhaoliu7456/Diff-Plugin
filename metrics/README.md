To calculate the FID and KID metrics, please make sure you have already installed the '*cleanfid*' package.

1. store the statistics offline
    ```python
    python store_statistics.py
    ```
2. calculate the FID/KID
    ```bash
    bash eval.sh
    ```

As the calculation speed of FID and KID is different, I split their calculation in two .py file