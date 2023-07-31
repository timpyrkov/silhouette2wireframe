[![Python Versions](https://img.shields.io/pypi/pyversions/wireframe?style=plastic)](https://pypi.org/project/wireframe/)
[![PyPI](https://img.shields.io/pypi/v/wireframe?style=plastic)](https://pypi.org/project/wireframe/)
[![License](https://img.shields.io/pypi/l/wireframe?style=plastic)](https://opensource.org/licenses/MIT)

# silhouette2wireframe

## Convert silhouette image to wireframe distorted by Perlin noise
#

# Installation
```
pip install wireframe
```


# Convert silhouette image to a wireframe

```
from wireframe import ImageToWireframe

converter = ImageToWireframe("silhouette.png", size=(1080,1350))

colors = {"bg": "#000022", "top": "dodgerblue", "bottom": "white"} 
fig = converter.draw_frame(0, cdict=colors)
plt.show()
```

![](img/silhouette.png)
![](img/arrow.png)
![](img/wireframe.jpg)

