# 데이터 파일

```
.
|-- toy-ppi-G.json
|-- toy-ppi-class_map.json
|-- toy-ppi-feats.npy
|-- toy-ppi-id_map.json
`-- toy-ppi-walks.txt
```

## toy-ppi 의 그래프 정보

* Undirected graph
* Node 수: 14755
* 피쳐 dimension: 50
* Edge 수: 228431
* Label 수: 121 (multi-label)
* Walks pair: 1895817

train, test, val 로 나누어져있으며 NetworkX의 그래프로 포팅하여 사용.
