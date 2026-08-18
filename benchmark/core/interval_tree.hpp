#pragma once

#include <vector>

namespace benchmark_core {

// A single stored interval and its associated payload.
struct IntervalEntry {
    int begin = 0; // inclusive
    int end = 0;   // exclusive (half-open, like Python's `intervaltree`)
    int data = 0;
};

// A dynamic augmented interval tree (a treap keyed on (begin, end, data),
// augmented with subtree-max-end for O(log n) expected overlap queries).
// Stands in for the Python `intervaltree` package used by blocks.py, which
// needs insert, exact-match remove, and overlap query on a structure that
// mutates during a sweep (so a static/offline tree won't do).
class IntervalTree {
public:
    IntervalTree() = default;
    ~IntervalTree();

    IntervalTree(const IntervalTree&) = delete;
    IntervalTree& operator=(const IntervalTree&) = delete;
    IntervalTree(IntervalTree&& other) noexcept;
    IntervalTree& operator=(IntervalTree&& other) noexcept;

    // Insert the half-open interval [begin, end) with payload `data`.
    void insert(int begin, int end, int data);

    // Remove the exact interval [begin, end) with payload `data`.
    // Returns false if no such entry was found.
    bool remove(int begin, int end, int data);

    // Return every stored interval overlapping the half-open range
    // [begin, end).
    std::vector<IntervalEntry> query(int begin, int end) const;

private:
    struct Node {
        int begin;
        int end;
        int data;
        int max_end;
        unsigned long long priority;
        Node* left = nullptr;
        Node* right = nullptr;

        Node(int b, int e, int d, unsigned long long p)
            : begin(b), end(e), data(d), max_end(e), priority(p) {}
    };

    Node* root_ = nullptr;

    static void update(Node* node);
    static Node* merge(Node* a, Node* b);
    static void split(Node* node, int begin, int end, int data, Node*& left, Node*& right);
    static Node* erase(Node* node, int begin, int end, int data, bool& removed);
    static void collect(const Node* node, int begin, int end, std::vector<IntervalEntry>& out);
    static void destroy(Node* node);
};

} // namespace benchmark_core
