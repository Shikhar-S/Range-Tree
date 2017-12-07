# Range-Tree
Pointerless implementation of Range Tree.
Works by first sorting using Bitonic Sort. Graph below shows the execution time of CPU(Xeon E5) and GPU(Quadro K620) algorithms. 
![Bitonic Sort Comparison](https://github.com/Shikhar-S/Range-Tree/blob/master/Images/Sorting_comparison.jpeg)
And Then merging the sorted arrays to form Secondary Trees using Merge Path Algorithm. 
![Merge Path vs CPU Merge](https://github.com/Shikhar-S/Range-Tree/blob/master/Images/temp.png)
The graph here shows combined results.
![Final Result](https://github.com/Shikhar-S/Range-Tree/blob/master/Images/RangeTreeConstruction.jpeg)
