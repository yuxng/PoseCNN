//
// (c) Matthew B. Kennel, Institute for Nonlinear Science, UCSD (2004)
//
// Licensed under the Academic Free License version 1.1 found in file LICENSE
// with additional provisions in that same file.


#include "kdtree.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>

namespace kdtree {
// utility
  
  inline float squared(const float x) {
    return(x*x);
  }
  
  inline void swap(int& a, int&b) {
    int tmp;
    tmp = a;
    a = b;
    b = tmp;
  }
  
  inline void swap(float& a, float&b) {
    float tmp;
    tmp = a;
    a = b;
    b = tmp;
  }
  
  //
  //       KDTREERESULT implementation
  //
  inline bool operator<(const KDTreeResult& e1, const KDTreeResult& e2) {
    return (e1.dis < e2.dis);
  }
  
  //
  //       KDTREE2_RESULT_VECTOR implementation
  //
  float KDTreeResultVector::max_value() {
    return( (*begin()).dis ); // very first element
  }
  
  void KDTreeResultVector::push_element_and_heapify(KDTreeResult& e) {
    push_back(e); // what a vector does.
    push_heap( begin(), end() ); // and now heapify it, with the new elt.
  }
  
  float KDTreeResultVector::replace_maxpri_elt_return_new_maxpri(KDTreeResult& e) {
    // remove the maximum priority element on the queue and replace it
    // with 'e', and return its priority.
    //
    // here, it means replacing the first element [0] with e, and re heapifying.
    
    pop_heap( begin(), end() );
    pop_back();
    push_back(e); // insert new
    push_heap(begin(), end() );  // and heapify.
    return( (*this)[0].dis );
  }
  
  //
  //        KDTREE2 implementation
  //

  // constructor
  KDTree::KDTree(KDTreeArray& data_in, bool rearrange_in, int dim_in)
  : the_data(data_in),
          N  ( data_in.shape()[0] ),
          dim( data_in.shape()[1] ),
          sort_results(false),
          rearrange(rearrange_in),
          root(NULL),
          data(NULL),
          ind(N) {
    //
    // initialize the constant references using this unusual C++
    // feature.
    //
    if (dim_in > 0)
      dim = dim_in;
    
    build_tree();
    
    if (rearrange) {
      // if we have a rearranged tree.
      // allocate the memory for it.
      rearranged_data.resize( boost::extents[N][dim] );
      
      // permute the data for it.
      for (int i=0; i<N; i++) {
        for (int j=0; j<dim; j++) {
          rearranged_data[i][j] = the_data[ind[i]][j];
          // wouldn't F90 be nice here?
        }
      }
      data = &rearranged_data;
    } else {
      data = &the_data;
    }
  }
  
  // destructor
  KDTree::~KDTree() {
    delete root;
  }
  
  // building routines
  void KDTree::build_tree() {
    for (int i=0; i<N; i++) ind[i] = i;
    root = build_tree_for_range(0, N-1, NULL);
  }
  
  KDTreeNode* KDTree::build_tree_for_range(int l, int u, KDTreeNode* parent) {
    
    // recursive function to build
    KDTreeNode* node = new KDTreeNode(dim);
    // the newly created node.
    
    if (u<l) {
      return(NULL); // no data in this node.
    }
    
    if ((u-l) <= bucketsize) {
      // create a terminal node.
      
      // always compute true bounding box for terminal node.
      for (int i=0;i<dim;i++) {
        spread_in_coordinate(i, l, u, node->box[i]);
      }
      
      node->cut_dim = 0;
      node->cut_val = 0.0;
      node->l = l;
      node->u = u;
      node->left = node->right = NULL;
      
      
    } else {
      //
      // Compute an APPROXIMATE bounding box for this node.
      // if parent == NULL, then this is the root node, and
      // we compute for all dimensions.
      // Otherwise, we copy the bounding box from the parent for
      // all coordinates except for the parent's cut dimension.
      // That, we recompute ourself.
      //
      int c = -1;
      float maxspread = 0.0;
      int m;
      
      for (int i=0;i<dim;i++) {
        if ((parent == NULL) || (parent->cut_dim == i)) {
          spread_in_coordinate(i, l, u, node->box[i]);
        } else {
          node->box[i] = parent->box[i];
        }
        float spread = node->box[i].upper - node->box[i].lower;
        if (spread>maxspread) {
          maxspread = spread;
          c=i;
        }
      }
      
      //
      // now, c is the identity of which coordinate has the greatest spread
      //
      
      if (false) {
        m = (l+u)/2;
        select_on_coordinate(c, m, l, u);
      } else {
        float sum;
        float average;
        
        if (true) {
          sum = 0.0;
          for (int k=l; k <= u; k++) {
            sum += the_data[ind[k]][c];
          }
          average = sum / static_cast<float> (u-l+1);
        } else {
          // average of top and bottom nodes.
          average = (node->box[c].upper + node->box[c].lower)*0.5F;
        }
        
        m = select_on_coordinate_value(c, average, l, u);
      }
      
      
      // move the indices around to cut on dim 'c'.
      node->cut_dim=c;
      node->l = l;
      node->u = u;
      
      node->left = build_tree_for_range(l, m, node);
      node->right = build_tree_for_range(m+1, u, node);
      
      if (node->right == NULL) {
        for (int i=0; i<dim; i++)
          node->box[i] = node->left->box[i];
        node->cut_val = node->left->box[c].upper;
        node->cut_val_left = node->cut_val_right = node->cut_val;
      } else if (node->left == NULL) {
        for (int i=0; i<dim; i++)
          node->box[i] = node->right->box[i];
        node->cut_val =  node->right->box[c].upper;
        node->cut_val_left = node->cut_val_right = node->cut_val;
      } else {
        node->cut_val_right = node->right->box[c].lower;
        node->cut_val_left  = node->left->box[c].upper;
        node->cut_val = (node->cut_val_left + node->cut_val_right) / 2.0F;
        //
        // now recompute true bounding box as union of subtree boxes.
        // This is now faster having built the tree, being logarithmic in
        // N, not linear as would be from naive method.
        //
        for (int i=0; i<dim; i++) {
          node->box[i].upper = std::max(node->left->box[i].upper,
                  node->right->box[i].upper);
          
          node->box[i].lower = std::min(node->left->box[i].lower,
                  node->right->box[i].lower);
        }
      }
    }
    return(node);
  }
  
  void KDTree::spread_in_coordinate(int c, int l, int u, interval& interv) {
    // return the minimum and maximum of the indexed data between l and u in
    // smin_out and smax_out.
    
    float smin, smax;
    float lmin, lmax;
    int i;
    
    smin = the_data[ind[l]][c];
    smax = smin;
    
    // process two at a time.
    for (i=l+2; i<= u; i+=2) {
      lmin = the_data[ind[i-1]] [c];
      lmax = the_data[ind[i]  ] [c];
      
      if (lmin > lmax) {
        swap(lmin, lmax);
        //      float t = lmin;
        //      lmin = lmax;
        //      lmax = t;
      }
      
      if (smin > lmin) smin = lmin;
      if (smax <lmax) smax = lmax;
    }
    // is there one more element?
    if (i == u+1) {
      float last = the_data[ind[u]] [c];
      if (smin>last) smin = last;
      if (smax<last) smax = last;
    }
    interv.lower = smin;
    interv.upper = smax;
  }
  
  
  void KDTree::select_on_coordinate(int c, int k, int l, int u) {
    //
    //  Move indices in ind[l..u] so that the elements in [l .. k]
    //  are less than the [k+1..u] elmeents, viewed across dimension 'c'.
    //
    while (l < u) {
      int t = ind[l];
      int m = l;
      
      for (int i=l+1; i<=u; i++) {
        if ( the_data[ ind[i] ] [c] < the_data[t][c]) {
          m++;
          swap(ind[i], ind[m]);
        }
      } // for i
      swap(ind[l], ind[m]);
      
      if (m <= k) l = m+1;
      if (m >= k) u = m-1;
    } // while loop
  }
  
  int KDTree::select_on_coordinate_value(int c, float alpha, int l, int u) {
    //
    //  Move indices in ind[l..u] so that the elements in [l .. return]
    //  are <= alpha, and hence are less than the [return+1..u]
    //  elments, viewed across dimension 'c'.
    //
    int lb = l, ub = u;
    
    while (lb < ub) {
      if (the_data[ind[lb]][c] <= alpha) {
        lb++; // good where it is.
      } else {
        swap(ind[lb], ind[ub]);
        ub--;
      }
    }
    
    // here ub=lb
    if (the_data[ind[lb]][c] <= alpha)
      return(lb);
    else
      return(lb-1);
    
  }
  
  // search record substructure
  //
  // one of these is created for each search.
  // this holds useful information to be used
  // during the search
  
  static const float infinity = 1.0e38F;
  
  class SearchRecord {
    
  private:
    friend class KDTree;
    friend class KDTreeNode;
    
    std::vector<float>& qv;
    int dim;
    bool rearrange;
    unsigned int nn; // , nfound;
    float ballsize;
    int centeridx, correltime;
    
    KDTreeResultVector& result;  // results
    const KDTreeArray* data;
    const std::vector<int>& ind;
    // constructor
    
  public:
    SearchRecord(std::vector<float>& qv_in, KDTree& tree_in,
            KDTreeResultVector& result_in) :
              qv(qv_in),
                      result(result_in),
                      data(tree_in.data),
                      ind(tree_in.ind) {
                dim = tree_in.dim;
                rearrange = tree_in.rearrange;
                ballsize = infinity;
                nn = 0;
              };
              
  };
  
  void KDTree::n_nearest_brute_force(std::vector<float>& qv, int nn, KDTreeResultVector& result) {
    
    result.clear();
    
    for (int i=0; i<N; i++) {
      float dis = 0.0;
      KDTreeResult e;
      for (int j=0; j<dim; j++) {
        dis += squared( the_data[i][j] - qv[j]);
      }
      e.dis = dis;
      e.idx = i;
      result.push_back(e);
    }
    sort(result.begin(), result.end() );
  }
  
  
  void KDTree::n_nearest(std::vector<float>& qv, int nn, KDTreeResultVector& result) {
    SearchRecord sr(qv, *this, result);
    std::vector<float> vdiff(dim, 0.0);
    
    result.clear();
    
    sr.centeridx = -1;
    sr.correltime = 0;
    sr.nn = nn;
    
    root->search(sr);
    
    if (sort_results) sort(result.begin(), result.end()); 
  }  
  
  void KDTree::n_nearest_around_point(int idxin, int correltime, int nn,
          KDTreeResultVector& result) {
    
    std::vector<float> qv(dim); //  query vector
    result.clear();
    for (int i=0; i<dim; i++) {
      qv[i] = the_data[idxin][i];
    }
    // copy the query vector.
    
    {
      SearchRecord sr(qv, *this, result);
      // construct the search record.
      sr.centeridx = idxin;
      sr.correltime = correltime;
      sr.nn = nn;
      root->search(sr);
    }
    
    if (sort_results) sort(result.begin(), result.end());
  }
  
  
  void KDTree::r_nearest(std::vector<float>& qv, float r2, KDTreeResultVector& result) {
  // search for all within a ball of a certain radius
    SearchRecord sr(qv, *this, result);
    std::vector<float> vdiff(dim, 0.0);
    
    result.clear();
    
    sr.centeridx = -1;
    sr.correltime = 0;
    sr.nn = 0;
    sr.ballsize = r2;
    
    root->search(sr);
    
    if (sort_results) sort(result.begin(), result.end());
  }
  
  int KDTree::r_count(std::vector<float>& qv, float r2) {
  // search for all within a ball of a certain radius
    {
      KDTreeResultVector result;
      SearchRecord sr(qv, *this, result);
      
      sr.centeridx = -1;
      sr.correltime = 0;
      sr.nn = 0;
      sr.ballsize = r2;
      
      root->search(sr);
      return(result.size());
    }
  }
  
  void KDTree::r_nearest_around_point(int idxin, int correltime, float r2,
          KDTreeResultVector& result) {
    std::vector<float> qv(dim);  //  query vector
    
    result.clear();
    
    for (int i=0; i<dim; i++) {
      qv[i] = the_data[idxin][i];
    }
    // copy the query vector.
    
    {
      SearchRecord sr(qv, *this, result);
      // construct the search record.
      sr.centeridx = idxin;
      sr.correltime = correltime;
      sr.ballsize = r2;
      sr.nn = 0;
      root->search(sr);
    }
    if (sort_results) sort(result.begin(), result.end());
  }
  
  int KDTree::r_count_around_point(int idxin, int correltime, float r2) {
    std::vector<float> qv(dim);  //  query vector
    
    for (int i=0; i<dim; i++) {
      qv[i] = the_data[idxin][i];
    }
    // copy the query vector.
    
    {
      KDTreeResultVector result;
      SearchRecord sr(qv, *this, result);
      // construct the search record.
      sr.centeridx = idxin;
      sr.correltime = correltime;
      sr.ballsize = r2;
      sr.nn = 0;
      root->search(sr);
      return(result.size());
    }
  }
  
  //
  //        KDTREE_NODE implementation
  //

  // constructor
  KDTreeNode::KDTreeNode(int dim) : box(dim) {
    left = right = NULL;
    //
    // all other construction is handled for real in the
    // KDTree building operations.
    //
  }
  
  // destructor
  KDTreeNode::~KDTreeNode() {
    if (left != NULL) delete left;
    if (right != NULL) delete right;
    // maxbox and minbox
    // will be automatically deleted in their own destructors.
  }
  
  
  void KDTreeNode::search(SearchRecord& sr) {
    // the core search routine.
    // This uses true distance to bounding box as the
    // criterion to search the secondary node.
    //
    // This results in somewhat fewer searches of the secondary nodes
    // than 'search', which uses the vdiff vector,  but as this
    // takes more computational time, the overall performance may not
    // be improved in actual run time.
    
    if ( (left == NULL) && (right == NULL)) {
      // we are on a terminal node
      if (sr.nn == 0) {
        process_terminal_node_fixedball(sr);
      } else {
        process_terminal_node(sr);
      }
    } else {
      KDTreeNode *ncloser, *nfarther;
      
      float extra;
      float qval = sr.qv[cut_dim];
      // value of the wall boundary on the cut dimension.
      if (qval < cut_val) {
        ncloser = left;
        nfarther = right;
        extra = cut_val_right-qval;
      } else {
        ncloser = right;
        nfarther = left;
        extra = qval-cut_val_left;
      };
      
      if (ncloser != NULL) ncloser->search(sr);
      
      if ((nfarther != NULL) && (squared(extra) < sr.ballsize)) {
        // first cut
        if (nfarther->box_in_search_range(sr)) {
          nfarther->search(sr);
        }
      }
    }
  }

  inline float dis_from_bnd(float x, float amin, float amax) {
    if (x > amax) {
      return(x-amax);
    } else if (x < amin)
      return (amin-x);
    else
      return 0.0;
  }
  
  inline bool KDTreeNode::box_in_search_range(SearchRecord& sr) {
    // does the bounding box, represented by minbox[*],maxbox[*]
    // have any point which is within 'sr.ballsize' to 'sr.qv'??
    
    int dim = sr.dim;
    float dis2 =0.0;
    float ballsize = sr.ballsize;
    for (int i=0; i<dim;i++) {
      dis2 += squared(dis_from_bnd(sr.qv[i], box[i].lower, box[i].upper));
      if (dis2 > ballsize)
        return(false);
    }
    return(true);
  }
  
  void KDTreeNode::process_terminal_node(SearchRecord& sr) {
    int centeridx  = sr.centeridx;
    int correltime = sr.correltime;
    unsigned int nn = sr.nn;
    int dim        = sr.dim;
    float ballsize = sr.ballsize;
    //
    bool rearrange = sr.rearrange;
    const KDTreeArray& data = *sr.data;
    
    const bool debug = false;
    
    if (debug) {
      printf("Processing terminal node %d, %d\n", l, u);
      std::cout << "Query vector = [";
      for (int i=0; i<dim; i++) std::cout << sr.qv[i] << ',';
      std::cout << "]\n";
      std::cout << "nn = " << nn << '\n';
      check_query_in_bound(sr);
    }
    
    for (int i=l; i<=u;i++) {
      int indexofi;  // sr.ind[i];
      float dis;
      bool early_exit;
      
      if (rearrange) {
        early_exit = false;
        dis = 0.0;
        for (int k=0; k<dim; k++) {
          dis += squared(data[i][k] - sr.qv[k]);
          if (dis > ballsize) {
            early_exit=true;
            break;
          }
        }
        if(early_exit) continue; // next iteration of mainloop
        // why do we do things like this?  because if we take an early
        // exit (due to distance being too large) which is common, then
        // we need not read in the actual point index, thus saving main
        // memory bandwidth.  If the distance to point is less than the
        // ballsize, though, then we need the index.
        //
        indexofi = sr.ind[i];
      } else {
        //
        // but if we are not using the rearranged data, then
        // we must always
        indexofi = sr.ind[i];
        early_exit = false;
        dis = 0.0;
        for (int k=0; k<dim; k++) {
          dis += squared(data[indexofi][k] - sr.qv[k]);
          if (dis > ballsize) {
            early_exit= true;
            break;
          }
        }
        if(early_exit) continue; // next iteration of mainloop
      } // end if rearrange.
      
      if (centeridx > 0) {
        // we are doing decorrelation interval
        if (abs(indexofi-centeridx) < correltime) continue; // skip this point.
      }
      
      // here the point must be added to the list.
      //
      // two choices for any point.  The list so far is either
      // undersized, or it is not.
      //
      if (sr.result.size() < nn) {
        KDTreeResult e;
        e.idx = indexofi;
        e.dis = dis;
        sr.result.push_element_and_heapify(e);
        if (debug) std::cout << "unilaterally pushed dis=" << dis;
        if (sr.result.size() == nn) ballsize = sr.result.max_value();
        // Set the ball radius to the largest on the list (maximum priority).
        if (debug) {
          std::cout << " ballsize = " << ballsize << "\n";
          std::cout << "sr.result.size() = "  << sr.result.size() << '\n';
        }
      } else {
        //
        // if we get here then the current node, has a squared
        // distance smaller
        // than the last on the list, and belongs on the list.
        //
        KDTreeResult e;
        e.idx = indexofi;
        e.dis = dis;
        ballsize = sr.result.replace_maxpri_elt_return_new_maxpri(e);
        if (debug) {
          std::cout << "Replaced maximum dis with dis=" << dis <<
                  " new ballsize =" << ballsize << '\n';
        }
      }
    } // main loop
    sr.ballsize = ballsize;
  }
  
  void KDTreeNode::process_terminal_node_fixedball(SearchRecord& sr) {
    int centeridx  = sr.centeridx;
    int correltime = sr.correltime;
    int dim        = sr.dim;
    float ballsize = sr.ballsize;
    //
    bool rearrange = sr.rearrange;
    const KDTreeArray& data = *sr.data;
    
    for (int i=l; i<=u;i++) {
      int indexofi = sr.ind[i];
      float dis;
      bool early_exit;
      
      if (rearrange) {
        early_exit = false;
        dis = 0.0;
        for (int k=0; k<dim; k++) {
          dis += squared(data[i][k] - sr.qv[k]);
          if (dis > ballsize) {
            early_exit=true;
            break;
          }
        }
        if(early_exit) continue; // next iteration of mainloop
        // why do we do things like this?  because if we take an early
        // exit (due to distance being too large) which is common, then
        // we need not read in the actual point index, thus saving main
        // memory bandwidth.  If the distance to point is less than the
        // ballsize, though, then we need the index.
        //
        indexofi = sr.ind[i];
      } else {
        //
        // but if we are not using the rearranged data, then
        // we must always
        indexofi = sr.ind[i];
        early_exit = false;
        dis = 0.0;
        for (int k=0; k<dim; k++) {
          dis += squared(data[indexofi][k] - sr.qv[k]);
          if (dis > ballsize) {
            early_exit= true;
            break;
          }
        }
        if(early_exit) continue; // next iteration of mainloop
      } // end if rearrange.
      
      if (centeridx > 0) {
        // we are doing decorrelation interval
        if (abs(indexofi-centeridx) < correltime) continue; // skip this point.
      }
      
      {
        KDTreeResult e;
        e.idx = indexofi;
        e.dis = dis;
        sr.result.push_back(e);
      }
      
    }
  }
} // namespace kdtee
