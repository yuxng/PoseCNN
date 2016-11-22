#ifndef __KDTREE_HPP
#define __KDTREE_HPP

// (c) Matthew B. Kennel, Institute for Nonlinear Science, UCSD (2004)
//
// Licensed under the Academic Free License version 1.1 found in file LICENSE
// with additional provisions in that same file.
//
// Implement a kd tree for fast searching of points in a fixed data base
// in k-dimensional Euclidean space.

#include <vector>
#include <algorithm>

#include <boost/multi_array.hpp>
#include <boost/array.hpp>

namespace kdtree {
  
  typedef boost::multi_array<float, 2>           KDTreeArray;
  typedef boost::const_multi_array_ref<float, 2> KDTreeROArray;
  
  typedef struct {
    float lower, upper;
  } interval;
  
  // let the compiler know that this is a names of classes.
  class KDTreeNode;
  class SearchRecord;  
  
  struct KDTreeResult {
  public:
    float dis;  // square distance
    int idx;    // neighbor index
  };
  
  class KDTreeResultVector : public std::vector<KDTreeResult> {
    // inherit a std::vector<KDTreeResult>
    // but, optionally maintain it in heap form as a priority
    // queue.
  public:
    
    //
    // add one new element to the list of results, and
    // keep it in heap order.  To keep it in ordinary, as inserted,
    // order, then simply use push_back() as inherited
    // via std::vector<>
    
    void push_element_and_heapify(KDTreeResult&);
    float replace_maxpri_elt_return_new_maxpri(KDTreeResult&);
    
    float max_value();
    // return the distance which has the maximum value of all on list,
    // assuming that ALL insertions were made by
    // push_element_and_heapify()
  };
  
  class KDTree {
  public:
    const KDTreeArray& the_data;
    // "the_data" is a reference to the underlying multi_array of the
    // data to be included in the tree.
    //
    // NOTE: this structure does *NOT* own the storage underlying this.
    // Hence, it would be a very bad idea to change the underlying data
    // during use of the search facilities of this tree.
    // Also, the user must deallocate the memory underlying it.
    
    
    const int N;          // number of data points
    int dim;
    bool sort_results;    // sorting result?
    const bool rearrange; // are we rearranging?
    
  public:

    // constructor, has optional 'dim_in' feature, to use only
    // first 'dim_in' components for definition of nearest neighbors.    
    KDTree(KDTreeArray& data_in, bool rearrange_in = true, int dim_in=-1);
    
    // destructor
    ~KDTree();
    
    
  public:
    
    void n_nearest_brute_force(std::vector<float>& qv, int nn, KDTreeResultVector& result);
    // search for n nearest to a given query vector 'qv' usin
    // exhaustive slow search.  For debugging, usually.
    
    void n_nearest(std::vector<float>& qv, int nn, KDTreeResultVector& result);
    // search for n nearest to a given query vector 'qv'.
    
    void n_nearest_around_point(int idxin, int correltime, int nn,
            KDTreeResultVector& result);
    // search for 'nn' nearest to point [idxin] of the input data, excluding
    // neighbors within correltime
    
    void r_nearest(std::vector<float>& qv, float r2, KDTreeResultVector& result);
    // search for all neighbors in ball of size (square Euclidean distance)
    // r2. Return number of neighbors in 'result.size()',
    
    void r_nearest_around_point(int idxin, int correltime, float r2,
            KDTreeResultVector& result);
    // like 'r_nearest', but around existing point, with decorrelation
    // interval.
    
    int r_count(std::vector<float>& qv, float r2);
    // count number of neighbors within square distance r2.
    
    int r_count_around_point(int idxin, int correltime, float r2);
    // like r_count, c
    
    friend class KDTreeNode;
    friend class SearchRecord;
    
  private:
    
    KDTreeNode* root; // the root pointer
    
    const KDTreeArray* data;
    // pointing either to the_data or an internal
    // rearranged data as necessary
    
    std::vector<int> ind;
    // the index for the tree leaves.  Data in a leaf with bounds [l,u] are
    // in  'the_data[ind[l],*] to the_data[ind[u],*]
    
    KDTreeArray rearranged_data;
    // if rearrange is true then this is the rearranged data storage.

    static const int bucketsize = 12;  // global constant.
    
  private:
    void set_data(KDTreeArray& din);
    void build_tree(); // builds the tree.  Used upon construction.
    KDTreeNode* build_tree_for_range(int l, int u, KDTreeNode* parent);
    void select_on_coordinate(int c, int k, int l, int u);
    int select_on_coordinate_value(int c, float alpha, int l, int u);
    void spread_in_coordinate(int c, int l, int u, interval& interv);
  };
  
  class KDTreeNode {
  public:
    KDTreeNode(int dim);
    ~KDTreeNode();
    
  private:
    // visible to self and KDTree.
    friend class KDTree;  // allow kdtree to access private data
    
    int cut_dim;                                 // dimension to cut;
    float cut_val, cut_val_left, cut_val_right;  //cut value
    int l, u;  // extents in index array for searching
    
    std::vector<interval> box; // [min,max] of the box enclosing all points
    
    KDTreeNode *left, *right;  // pointers to left and right nodes.
    
    void search(SearchRecord& sr);
    // recursive innermost core routine for searching..
    
    bool box_in_search_range(SearchRecord& sr);
    // return true if the bounding box for this node is within the
    // search range given by the searchvector and maximum ballsize in 'sr'.
    
    void check_query_in_bound(SearchRecord& sr); // debugging only
    
    // for processing final buckets.
    void process_terminal_node(SearchRecord& sr);
    void process_terminal_node_fixedball(SearchRecord& sr); 
  };
  
} // namespace kdtree

#endif
