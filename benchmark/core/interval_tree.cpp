#include "interval_tree.hpp"

#include <algorithm>
#include <random>
#include <tuple>

namespace benchmark_core {

namespace {

unsigned long long next_priority() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng();
}

bool triple_less(int b1, int e1, int d1, int b2, int e2, int d2) {
    return std::tie(b1, e1, d1) < std::tie(b2, e2, d2);
}

} // namespace

IntervalTree::~IntervalTree() {
    destroy(root_);
    root_ = nullptr;
    // Free-list nodes are real heap objects too (just detached from the
    // tree), so they need the same cleanup.
    while (free_list_) {
        Node* next = free_list_->left;
        delete free_list_;
        free_list_ = next;
    }
}

IntervalTree::IntervalTree(IntervalTree&& other) noexcept
    : root_(other.root_), free_list_(other.free_list_) {
    other.root_ = nullptr;
    other.free_list_ = nullptr;
}

IntervalTree& IntervalTree::operator=(IntervalTree&& other) noexcept {
    if (this != &other) {
        destroy(root_);
        while (free_list_) {
            Node* next = free_list_->left;
            delete free_list_;
            free_list_ = next;
        }
        root_ = other.root_;
        free_list_ = other.free_list_;
        other.root_ = nullptr;
        other.free_list_ = nullptr;
    }
    return *this;
}

void IntervalTree::update(Node* node) {
    if (!node) {
        return;
    }
    node->max_end = node->end;
    if (node->left) {
        node->max_end = std::max(node->max_end, node->left->max_end);
    }
    if (node->right) {
        node->max_end = std::max(node->max_end, node->right->max_end);
    }
}

// Standard treap merge: assumes every key in `a` is less than every key in
// `b`. Combines them while preserving the max-heap property on priority.
IntervalTree::Node* IntervalTree::merge(Node* a, Node* b) {
    if (!a) return b;
    if (!b) return a;
    if (a->priority > b->priority) {
        a->right = merge(a->right, b);
        update(a);
        return a;
    }
    b->left = merge(a, b->left);
    update(b);
    return b;
}

// Splits `node`'s subtree into `left` (keys strictly less than
// (begin,end,data)) and `right` (keys greater or equal).
void IntervalTree::split(Node* node, int begin, int end, int data, Node*& left, Node*& right) {
    if (!node) {
        left = right = nullptr;
        return;
    }
    if (triple_less(node->begin, node->end, node->data, begin, end, data)) {
        split(node->right, begin, end, data, node->right, right);
        left = node;
        update(left);
    } else {
        split(node->left, begin, end, data, left, node->left);
        right = node;
        update(right);
    }
}

IntervalTree::Node* IntervalTree::alloc_node(int begin, int end, int data,
                                              unsigned long long priority) {
    if (free_list_) {
        Node* node = free_list_;
        free_list_ = free_list_->left;
        node->begin = begin;
        node->end = end;
        node->data = data;
        node->max_end = end;
        node->priority = priority;
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    return new Node(begin, end, data, priority);
}

void IntervalTree::free_node(Node* node) {
    node->left = free_list_;
    free_list_ = node;
}

void IntervalTree::insert(int begin, int end, int data) {
    Node* node = alloc_node(begin, end, data, next_priority());
    Node* left = nullptr;
    Node* right = nullptr;
    split(root_, begin, end, data, left, right);
    root_ = merge(merge(left, node), right);
}

IntervalTree::Node* IntervalTree::erase(Node* node, int begin, int end, int data, bool& removed) {
    if (!node) {
        return nullptr;
    }
    if (node->begin == begin && node->end == end && node->data == data) {
        removed = true;
        Node* merged = merge(node->left, node->right);
        free_node(node);
        return merged;
    }
    if (triple_less(begin, end, data, node->begin, node->end, node->data)) {
        node->left = erase(node->left, begin, end, data, removed);
    } else {
        node->right = erase(node->right, begin, end, data, removed);
    }
    update(node);
    return node;
}

bool IntervalTree::remove(int begin, int end, int data) {
    bool removed = false;
    root_ = erase(root_, begin, end, data, removed);
    return removed;
}

void IntervalTree::collect(const Node* node, int begin, int end, std::vector<IntervalEntry>& out) {
    if (!node) {
        return;
    }
    if (node->left && node->left->max_end > begin) {
        collect(node->left, begin, end, out);
    }
    if (node->begin < end && begin < node->end) {
        out.push_back({node->begin, node->end, node->data});
    }
    if (node->begin < end) {
        collect(node->right, begin, end, out);
    }
}

std::vector<IntervalEntry> IntervalTree::query(int begin, int end) const {
    std::vector<IntervalEntry> out;
    collect(root_, begin, end, out);
    return out;
}

void IntervalTree::query(int begin, int end, std::vector<IntervalEntry>& out) const {
    collect(root_, begin, end, out);
}

void IntervalTree::destroy(Node* node) {
    if (!node) {
        return;
    }
    destroy(node->left);
    destroy(node->right);
    delete node;
}

} // namespace benchmark_core
