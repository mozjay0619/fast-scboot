

FastScboot
==========

FastScboot is a statistics tool to perform the stratified clustered bootstrap sampling on given data. The algorithm is fast in the sense that the remaining bottleneck to the speed of the algorithm is the speed of memory access during the inplace fancy indexing operation.

Install
-------

::

	pip install fast-scboot

Getting started
---------------



How does it work?
-----------------

.. image:: https://github.com/mozjay0619/fast-scboot/blob/master/media/img1.png
	:width: 600pt

When the ``prepare_data`` method is invoked, once the original data has been sorted by strata and cluster levels, the ``make_index_matrix`` creates three auxiliary arrays: ``idx_mtx``, ``strat_arr``, and ``clust_arr``. The ``idx_mtx`` array stores information on where each cluster begins and how many rows it occupies, as well as the actual cluster value. The ``strat_arr`` is an index array that indexes the strata levels at each of the cluster level. The ``clust_arr`` does the same but for the cluster levels. The reason the values of the ``clust_arr`` are not uniformly increasing like ``strat_arr`` in this example is because internally, the unique indices are created using the cantor pairing function for speed (and then re-cast into integer using Pandas "cateory" type).

When the ``sample_data`` method is invoked, three additional auxiliary data are created. The ``clust_cnt_arr`` array stores the number of unique cluster values in each strata, in this case, [3, 2, 2]. The total number of unique strata values is stored in the ``num_strats`` variable, and the same for cluster is store in the ``num_clusts`` variable.

