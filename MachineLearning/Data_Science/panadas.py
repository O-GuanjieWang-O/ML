import string

import numpy as np
import pandas as pd


def readTable():
    a = np.arange(16)
    v = pd.Series(a, index=list(string.ascii_uppercase[i] for i in range(16)))
    # get the value: v["A"]/ v.get("A")
    # print(v["A"])
    data = {
        "calories": [420, 380, 390],
        "duration": [50, 40, 45]
    }

    myvar = pd.DataFrame(data,index=list(string.ascii_uppercase[i] for i in range(3)))
    print(myvar)
    # loc the row
    print(myvar.loc["A"])


if __name__ == "__main__":
    readTable()
