#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;


class BinaryDecisionTree
{
public:
    struct TreeNode
    {
        bool isLeaf{false};
        TreeNode *leftChild{nullptr};
        TreeNode *rightChild{nullptr};
        int feature{0};
        double threshold{0.0};
        double gini{0.0};
        double n1{0.0};
        double n2{0.0};
        int label{0};
        int number{0};
    };
    
    void load_dataset(const string& csv_file_path);

    void print_dataset();

    pair<double, double> count_labels(const vector<size_t>& data_indices) const;

    vector<double> uniq_features(vector<size_t> data_indices, size_t feature_index);

    pair<vector<size_t>, vector<size_t>> split_data_indices(vector<size_t> data_indices, size_t feature_index, double temp_threshold);

    double gini(const vector<size_t>& data_indices) const;

    pair<double, double> best_threshold(vector<size_t> data_indices, size_t feature);

    TreeNode *build_tree(vector<size_t> data_indices, vector<size_t> feature_indices);
    
    void train();

    void collect_leaf_node(const TreeNode* node, double& correct_predictions, double& total_predictions);

    double accuracy();

    void traverse_tree(const TreeNode* node, ofstream& out);

    void generate_dot_file();

private:
    vector<vector<float>> x;
    vector<int> y;

    struct TreeNode* root;
};

/**
 * @brief Loads a dataset from a specified CSV file into the decision tree.
 * 
 * This function opens a CSV file specified by the `csv_file_path` parameter and reads its content line by line. 
 * The first line, assumed to be a header, is skipped. Each subsequent line is expected to contain a series 
 * of feature values followed by a label, all separated by commas. The function extracts these values, 
 * converting feature values to floats and the label to an integer. Each set of features is stored in 
 * a vector of floats (`std::vector<float>`), and all such vectors are collected into the member variable `x`. 
 * The labels are stored in the member variable `y`. The number of features per dataset entry is expected 
 * to be constant and is specified by the `featureCount` constant within the function.
 * 
 * The function uses `std::move` to transfer the features vector to the `x` member variable efficiently, 
 * without unnecessary copying. This operation leaves the moved-from vector in a valid but unspecified state.
 * 
 * @param csv_file_path The path to the CSV file to be loaded. The file must exist and be readable.
 * 
 * @throw std::runtime_error if the file specified by `csv_file_path` cannot be opened.
 * @throw std::invalid_argument if any line in the CSV file contains data that cannot be correctly parsed into floats (for features) or an int (for the label).
 */
void BinaryDecisionTree::load_dataset(const string& csv_file_path) {
    fstream file(csv_file_path, ios::in);

    if (!file.is_open()) {
        throw runtime_error("Could not open the file: " + csv_file_path);
    }

    string line;

    getline(file, line);

    const size_t featureCount = 7;

    while (getline(file, line)) {
        stringstream str(line);
        vector<float> features;
        features.reserve(featureCount);
        int label;
        string word;
        size_t count = 0;

        while (getline(str, word, ',')) {
            if (count < featureCount) {
                features.push_back(stof(word));
            } else if (count == featureCount) {
                label = stoi(word);
                break;
            }
            count++;
        }

        x.push_back(move(features));
        y.push_back(label);
    }

    file.close();
}

/**
 * @brief Prints the currently loaded dataset to the standard output.
 * 
 * This method iterates over the dataset stored in the class's `x` (features) and `y` (labels) member variables, 
 * printing each sample's features followed by its label. It's designed to provide a quick overview of the dataset's 
 * contents, including the total number of samples and the number of features per sample. This can be useful for 
 * verifying the correct loading of data or for debugging purposes.
 * 
 * Each sample's features are printed on a single line, prefixed with "x: ", followed by the label, prefixed with "y: ".
 * After listing all samples, the method prints the total number of samples and the assumed constant number of features 
 * per sample. Note that the method assumes the dataset is not empty, especially when printing the feature size, 
 * as it directly accesses the first sample with `x[0].size()`. If the dataset could be empty, consider checking 
 * `x.size()` before accessing `x[0]` to avoid potential access violations.
 */
void BinaryDecisionTree::print_dataset() {
    for (size_t i = 0; i < x.size(); ++i) {
        cout << "x: ";
        
        for (float feature : x[i]) {
            cout << feature << " ";
        }
        
        cout << "y: " << y[i] << endl;
    }
    cout << "Data size: " << x.size() << endl;
    cout << "Feature size: " << x[0].size() << endl;
}

/**
 * @brief Counts the occurrences of positive and negative labels within a subset of the dataset.
 * 
 * This method iterates over a given set of indices (`data_indices`), referring to the dataset stored in the 
 * class's `y` member variable (labels), and counts how many times positive (1) and negative (0) labels occur. 
 * This can be useful for assessing the balance of the dataset or for calculating metrics that depend on the 
 * distribution of labels in a subset of the data, such as the Gini impurity or entropy in decision tree nodes.
 * 
 * The method expects the indices in `data_indices` to be valid and within the range of the dataset. If an 
 * index is out of range or the label at an index is not 0 or 1, the method throws an exception.
 * 
 * @param data_indices A vector of indices for which labels should be counted.
 * @return A pair of doubles where the first element is the count of positive labels and the second element is the count of negative labels.
 * 
 * @throw std::out_of_range if any index in `data_indices` is out of the range of the dataset.
 * @throw std::runtime_error if an unexpected label (not 0 or 1) is encountered.
 */
pair<double, double> BinaryDecisionTree::count_labels(const vector<size_t>& data_indices) const {
        double positive = 0, negative = 0;

        for (auto index : data_indices) {
            if (index >= y.size()) {
                throw out_of_range("Index out of range.");
            }
            
            int label = y[index];
            switch(label) {
                case 0: negative++; break;
                case 1: positive++; break;
                default: throw runtime_error("Unexpected label encountered.");
            }
        }

        return {positive, negative};
}

/**
 * @brief Extracts and returns the unique values of a specified feature from a subset of the dataset.
 * 
 * This method iterates over the dataset using the indices provided in `data_indices` to focus on a specific subset. 
 * It then accesses the value of the feature specified by `feature_index` for each sample in the subset. The method 
 * collects each unique value of this feature, ensuring no duplicates are included in the final list. This functionality 
 * is crucial for decision tree algorithms during the process of finding the best split. Identifying unique values 
 * of a feature allows the algorithm to evaluate possible splitting points efficiently.
 * 
 * @param data_indices A vector of indices specifying the subset of the dataset to examine. These should be valid indices within the range of the dataset.
 * @param feature_index The index of the feature for which unique values are to be extracted. This should be within the range of features available in the dataset.
 * @return A vector of doubles, each representing a unique value of the specified feature found in the specified subset of the dataset.
 * 
 * @note This method does not sort the returned vector of unique values; they are returned in the order they are found.
 */
vector<double> BinaryDecisionTree::uniq_features(vector<size_t> data_indices, size_t feature_index) {
    vector<double> result;

    for (auto index : data_indices) {
        double value = x[index][feature_index];
        if (find(result.begin(), result.end(), value) == result.end()) {
            result.push_back(value);
        }
    }

    return result;
}

/**
 * @brief Splits a subset of the dataset into two groups based on a threshold value of a specified feature.
 * 
 * This method is pivotal for constructing binary decision trees. Given a set of data indices, a feature index, 
 * and a threshold value (`temp_threshold`), it divides the dataset into two subsets: one where the feature's value 
 * is less than or equal to the threshold (left child) and another where the feature's value is greater than the threshold (right child).
 * This binary split forms the basis for further recursive partitioning of the dataset in the decision tree construction process.
 * 
 * @param data_indices A vector of indices representing the subset of the dataset to be split. These indices should 
 * be valid within the full dataset.
 * @param feature_index The index of the feature used for the split. This index should correspond to one of the features 
 * within the dataset and is zero-based.
 * @param temp_threshold The threshold value for the feature at `feature_index`. Data points with a feature value 
 * less than or equal to this threshold will be placed in the left child subset, and those with a value greater 
 * than the threshold will be placed in the right child subset.
 * @return A pair of vectors of size_t, where the first vector contains indices of the dataset forming the left child subset,
 * and the second vector contains indices forming the right child subset.
 */
pair<vector<size_t>, vector<size_t>> BinaryDecisionTree::split_data_indices(vector<size_t> data_indices, size_t feature_index, double temp_threshold){
    vector<size_t> left_child_data_indices;
    vector<size_t> right_child_data_indices;
    
    for (auto index : data_indices)
    {
        if (x[index][feature_index] <= temp_threshold)
        {
            left_child_data_indices.push_back(index);
        }
        else
        {
            right_child_data_indices.push_back(index);
        }
    }

    return {left_child_data_indices, right_child_data_indices};
}

/**
 * @brief Calculates the Gini impurity of a subset of the dataset.
 * 
 * Gini impurity is a measure used in decision trees to quantify the likelihood of an incorrect classification 
 * if a random label was assigned according to the label distribution in the subset. A Gini impurity of 0 indicates 
 * that all elements in the subset belong to a single class, making it a measure of class purity.
 * 
 * This method iterates through the subset specified by `data_indices`, counting the occurrences of positive (1) 
 * and negative (0) labels to calculate the Gini impurity. The formula used is:
 *      1 - sum(pi^2)
 * where pi is the proportion of samples in class i (for this method, i ∈ {positive, negative}).
 * 
 * @param data_indices A vector of indices representing the subset of the dataset for which to calculate Gini impurity. 
 * These indices should be valid within the full dataset.
 * @return The Gini impurity of the subset, a double value between 0 (pure) and 0.5 (equally divided).
 * 
 * @throw std::out_of_range if any index in `data_indices` is out of the range of the dataset.
 * @throw std::runtime_error if an unexpected label (not 0 or 1) is encountered in the subset.
 */
double BinaryDecisionTree::gini(const vector<size_t>& data_indices) const {
        double positive = 0, negative = 0;

        for (auto index : data_indices) {
            if (index >= y.size()) {
                throw out_of_range("Index out of range.");
            }
            
            int label = y[index];
            switch(label) {
                case 0: negative++; break;
                case 1: positive++; break;
                default: throw runtime_error("Unexpected label encountered.");
            }
        }

        return 1 - pow(positive / data_indices.size(), 2) - pow(negative / data_indices.size(), 2);
}

/**
 * @brief Finds the best threshold for splitting a dataset on a specific feature to minimize Gini impurity.
 * 
 * This function iterates through all unique values of a given feature in a subset of the dataset, 
 * evaluating each value as a potential threshold for splitting the dataset into two parts. For each 
 * potential threshold, it calculates the weighted average Gini impurity of the two resulting subsets. 
 * The function selects the threshold that results in the lowest weighted average Gini impurity, subject 
 * to the constraint that each resulting subset contains at least a minimum number of elements (e.g., 5) 
 * to avoid overly granular splits.
 * 
 * The optimal threshold is determined through an exhaustive search among all unique feature values. This 
 * approach ensures that the selected threshold is the most effective among the considered values for 
 * reducing uncertainty (impurity) within the resulting subsets.
 * 
 * @param data_indices A vector of indices specifying the subset of the dataset to examine. These indices 
 * should be valid within the full dataset and refer to samples that include the feature of interest.
 * @param feature_index The index of the feature across which to find the best splitting threshold. This 
 * index should correspond to one of the features within the dataset.
 * @return A pair containing the best threshold (double) for the feature and the corresponding minimum average 
 * Gini impurity (double) achieved by this split. The first element of the pair is the threshold, and the second 
 * element is the Gini impurity.
 */
pair<double, double> BinaryDecisionTree::best_threshold(vector<size_t> data_indices, size_t feature_index)
{       
    vector<double> features = uniq_features(data_indices, feature_index);

    double min_avg_gini = 0.5;
    double threshold = features[0];


    for (auto temp_threshold : features)
    {   

        pair<vector<size_t>, vector<size_t>> sub_data_indices = split_data_indices(data_indices, feature_index, temp_threshold);


        double avg_gini = gini(sub_data_indices.first) * (double)sub_data_indices.first.size() / data_indices.size() + gini(sub_data_indices.second) * (double)sub_data_indices.second.size() / data_indices.size();
        
        if (avg_gini < min_avg_gini && sub_data_indices.first.size() >= 5 && sub_data_indices.second.size() >= 5)
        {
            min_avg_gini = avg_gini;
            threshold = temp_threshold;
        }
    }
    return {threshold, min_avg_gini};
}

/**
 * @brief Recursively builds the decision tree based on the given dataset and feature indices.
 * 
 * This function represents the core of the decision tree algorithm. Starting from a set of data indices and
 * feature indices, it recursively splits the data to construct the decision tree. At each node, it selects the
 * best feature and threshold for the split based on the criterion of minimizing the Gini impurity. The recursion
 * ends when either of the following conditions is met: all samples at the node have the same label, there are no
 * features left to split on, or the decrease in impurity is not significant.
 * 
 * The function dynamically allocates new nodes for the tree, determining whether each node should become a leaf
 * based on the stopping criteria. Leaf nodes are assigned a label based on the majority class in the subset. Each
 * node in the tree keeps track of the Gini impurity, the counts of positive and negative samples, the chosen
 * feature and threshold for the split (for non-leaf nodes), and whether it is a leaf node.
 * 
 * @param data_indices A vector of indices indicating the subset of the dataset that the current node is
 * responsible for.
 * @param feature_indices A vector of indices indicating the features that are still available for splitting
 * at the current node.
 * @return A pointer to the root of the subtree constructed by this function. This pointer points to a
 * dynamically allocated TreeNode, which the caller is responsible for managing.
 */
BinaryDecisionTree::TreeNode *BinaryDecisionTree::build_tree(vector<size_t> data_indices, vector<size_t> feature_indices)
{
    static int number = -1;
    number++;
    
    TreeNode *node = new TreeNode;

    pair<double, double> labels = count_labels(data_indices);
    node->n1 = labels.first;
    node->n2 = labels.second;
    node->gini = 1 - pow(node->n1 / data_indices.size(), 2) - pow(node->n2 / data_indices.size(), 2);
    node->number = number;
    cout << "Create node [" << number << "]. Number of n1: " << node->n1 << ". Number of n2: " << node->n2 << ". GINI: " << node->gini << endl;


    if (node->n1 == 0 || node->n2 == 0 || feature_indices.size() == 0)
    {
        node->isLeaf = true;
        node->label = (node->n1 >= node->n2) ? 1 : 0;
        return node;
    }

    int best_feature_index = feature_indices[0];
    double threshold = 0;
    double min_avg_gini = 0.5;
    int eraseIndex = 0;
    for (int i=0; i<feature_indices.size(); i++)
    {
        pair<double, double> threshold_ginisplit = best_threshold(data_indices, feature_indices[i]);
        if (threshold_ginisplit.second <= min_avg_gini) 
        {
            threshold = threshold_ginisplit.first;
            min_avg_gini = threshold_ginisplit.second;
            best_feature_index = feature_indices[i];
            eraseIndex = i;
        }
    }

    vector<size_t> sub_feature_indices(feature_indices);
    sub_feature_indices.erase(sub_feature_indices.begin() + eraseIndex);

    if (min_avg_gini < node->gini)
    {   
        pair<vector<size_t>, vector<size_t>> sub_data_indices = split_data_indices(data_indices, best_feature_index, threshold);

        cout<<"Left branch ";
        node->leftChild = build_tree(sub_data_indices.first, sub_feature_indices);
        cout<<"Right branch ";
        node->rightChild = build_tree(sub_data_indices.second, sub_feature_indices);
        node->feature = best_feature_index;
        node->threshold = threshold;
        return node;        
    }
    else
    {
        node->isLeaf = true;
        node->label = (node->n1 >= node->n2) ? 1 : 0;
        return node;
    }
}

/**
 * @brief Initiates the training process of the decision tree using the dataset loaded into the `x` and `y` member variables.
 * 
 * This method prepares for the construction of the decision tree by generating a list of indices for the dataset
 * and for the features. These indices are then used to build the tree. The method sets up the initial conditions
 * for building the tree by including all data points and all features as candidates for the first split. It then
 * calls the `build_tree` method, which recursively splits the dataset to construct the decision tree. The root of
 * the constructed tree is stored in the `root` member variable.
 * 
 * The `train` method effectively transforms the raw data loaded into the decision tree's structure into a model
 * that can be used for making predictions. It assumes that the dataset (`x` and `y`) has already been loaded and
 * is non-empty. This method does not return a value, but upon completion, the `BinaryDecisionTree` object will
 * contain the trained decision tree model accessible through the `root` pointer.
 */
void BinaryDecisionTree::train(){
    
    vector<size_t> data_indices;
    for (size_t i = 0; i < x.size(); i++) {
        data_indices.push_back(i);
    }

    vector<size_t> feature_indices;
    for (size_t i = 0; i < x[0].size(); i++) {
        feature_indices.push_back(i);
    }

    root = build_tree(data_indices, feature_indices);

}

/**
 * @brief Recursively traverses the decision tree to collect accuracy statistics from the leaf nodes.
 * 
 * This method is designed to evaluate the decision tree's performance by examining each leaf node's
 * prediction accuracy. For each leaf node, it calculates the number of instances that would be correctly
 * predicted based on the majority class within that node. The method accumulates these statistics across
 * all leaf nodes to provide an overall measure of the tree's accuracy.
 * 
 * The accuracy calculation assumes that the leaf node's prediction is the majority class label within that
 * node. The `correct_predictions` counter is incremented by the number of instances in the node that actually
 * belong to the majority class, and the `total_predictions` counter is incremented by the total number of
 * instances in the node.
 * 
 * This method is called recursively, starting from the tree's root node and proceeding down to the leaf nodes.
 * It updates the `correct_predictions` and `total_predictions` variables passed by reference, which should be
 * initialized to zero before the first call.
 * 
 * @param node The current tree node being examined. This should be the root node when the method is first called.
 * @param correct_predictions A reference to a double variable that accumulates the count of correctly predicted instances.
 * @param total_predictions A reference to a double variable that accumulates the total count of instances examined.
 */
void BinaryDecisionTree::collect_leaf_node(const TreeNode* node, double& correct_predictions, double& total_predictions) {
    if (!node) return;
    if (node->isLeaf) {
        double predictedLabel = node->label;
        double instances = node->n1 + node->n2;
        correct_predictions += (predictedLabel == 1) ? node->n1 : node->n2;
        total_predictions += instances;
        return;
    }
    collect_leaf_node(node->leftChild, correct_predictions, total_predictions);
    collect_leaf_node(node->rightChild, correct_predictions, total_predictions);
}

/**
 * @brief Calculates the accuracy of the decision tree model.
 * 
 * This method computes the decision tree's accuracy as the proportion of correctly predicted instances
 * to the total number of instances in the dataset, based on the tree's structure and the distribution of
 * labels in its leaf nodes. It calls the `collect_leaf_node` method starting from the root to aggregate
 * correct and total predictions across all leaf nodes of the tree.
 * 
 * The accuracy is a measure of the model's performance, indicating how often it correctly predicts the label
 * of an instance. This method assumes the decision tree has already been trained and the leaf nodes accurately
 * reflect the outcome of the decision process for subsets of the dataset they represent.
 * 
 * @return The accuracy of the model as a double, which is the ratio of correct predictions to total predictions.
 *         The value ranges from 0 (no correct predictions) to 1 (all predictions correct).
 */
double BinaryDecisionTree::accuracy() {
    double correct_predictions = 0.0;
    double total_predictions = 0.0;
    
    collect_leaf_node(root, correct_predictions, total_predictions);

    return correct_predictions / total_predictions;
}

/**
 * @brief Recursively traverses the decision tree and writes its structure to a DOT format file for visualization.
 * 
 * This method is essential for visualizing the decision tree structure, providing insights into how the tree
 * makes decisions based on the features. For each node in the tree, it outputs DOT language statements representing
 * the node and its connections to child nodes. Leaf nodes are represented as ellipses with their label and statistics,
 * while decision nodes are represented as boxes containing the feature and threshold used for splitting, along with
 * node statistics. This method is called recursively, starting from the root node, to ensure the entire tree is
 * processed.
 * 
 * The DOT file generated by this method can be rendered into an image using Graphviz tools, allowing for an
 * easy-to-understand graphical representation of the tree.
 * 
 * @param node The current node being visited. This should be the root node when the method is first called.
 * @param out A reference to an `std::ofstream` object opened for the DOT file where the tree structure will be written.
 */
void BinaryDecisionTree::traverse_tree(const TreeNode* node, std::ofstream& out) {
    if (!node) return;

    if (node->isLeaf) {
        out << "    " << node->number << " [label=\"Label: " << node->label << "\\nTotal: " << (int)(node->n1+node->n2) << "\\nn1: " << node->n1 << "\\nn2: " << node->n2 << "\\nGini: " << node->gini << "\", shape=ellipse];\n";
    } else {
        out << "    " << node->number << " [label=\"Feature " << node->feature << " <= " << node->threshold << "\\nTotal: " << (int)(node->n1+node->n2) << "\\nn1: " << node->n1 << "\\nn2: " << node->n2 << "\\nGini: " << node->gini << "\", shape=box];\n";
    }

    if (node->leftChild) {
        out << "    " << node->number << " -> " << node->leftChild->number << " [label=\"True\"];\n";
        traverse_tree(node->leftChild, out);
    }
    if (node->rightChild) {
        out << "    " << node->number << " -> " << node->rightChild->number << " [label=\"False\"];\n";
        traverse_tree(node->rightChild, out);
    }
}

/**
 * @brief Generates a DOT file representing the decision tree structure.
 * 
 * This method initiates the process of creating a visual representation of the decision tree by generating a DOT
 * file named "tree.dot". The DOT language is a graph description language that allows for the easy generation of
 * graphs through tools like Graphviz. The method sets up the initial structure of the DOT file, including graph
 * and node properties, and then calls `traverse_tree` to fill in the details of each node and their connections.
 * 
 * The output DOT file includes the decision nodes, leaf nodes, and edges that connect these nodes based on the
 * tree's branching logic. Decision nodes are detailed with the feature and threshold used for splitting, and leaf
 * nodes include the predicted label and statistics about the data points that reached that leaf. After the file
 * is generated, it can be rendered into an image using Graphviz, offering a clear visual overview of how the tree
 * operates.
 * 
 * The method concludes by closing the file and notifying the user of the file's creation through standard output.
 */
void BinaryDecisionTree::generate_dot_file() {
    ofstream out("tree.dot");
    out << "digraph Tree {\n";
    out << "    node [fontname=\"Helvetica\"];\n";
    
    if (root) {
        traverse_tree(root, out);
    }

    out << "}\n";
    out.close();
    cout << "DOT file generated.\n";
}

int main()
{   
    BinaryDecisionTree model;
    model.load_dataset("Diagnosis_7features.csv");
    model.print_dataset();
    model.train();
    double accuracy = model.accuracy();
    cout << "Accuracy: " << accuracy << endl;
    model.generate_dot_file();
    return 0;
}

/* Output
Create node [0]. Number of n1: 328. Number of n2: 312. GINI: 0.499688
Left branch Create node [1]. Number of n1: 133. Number of n2: 39. GINI: 0.350663
Left branch Create node [2]. Number of n1: 50. Number of n2: 5. GINI: 0.165289
Left branch Create node [3]. Number of n1: 3. Number of n2: 3. GINI: 0.5
Right branch Create node [4]. Number of n1: 47. Number of n2: 2. GINI: 0.0783007
Left branch Create node [5]. Number of n1: 24. Number of n2: 2. GINI: 0.142012
Left branch Create node [6]. Number of n1: 9. Number of n2: 0. GINI: 0
Right branch Create node [7]. Number of n1: 15. Number of n2: 2. GINI: 0.207612
Right branch Create node [8]. Number of n1: 23. Number of n2: 0. GINI: 0
Right branch Create node [9]. Number of n1: 83. Number of n2: 34. GINI: 0.412302
Left branch Create node [10]. Number of n1: 74. Number of n2: 24. GINI: 0.369846
Left branch Create node [11]. Number of n1: 16. Number of n2: 2. GINI: 0.197531
Right branch Create node [12]. Number of n1: 58. Number of n2: 22. GINI: 0.39875
Right branch Create node [13]. Number of n1: 9. Number of n2: 10. GINI: 0.498615
Left branch Create node [14]. Number of n1: 5. Number of n2: 9. GINI: 0.459184
Right branch Create node [15]. Number of n1: 4. Number of n2: 1. GINI: 0.32
Right branch Create node [16]. Number of n1: 195. Number of n2: 273. GINI: 0.486111
Left branch Create node [17]. Number of n1: 115. Number of n2: 87. GINI: 0.490393
Left branch Create node [18]. Number of n1: 70. Number of n2: 32. GINI: 0.430604
Left branch Create node [19]. Number of n1: 11. Number of n2: 14. GINI: 0.4928
Right branch Create node [20]. Number of n1: 59. Number of n2: 18. GINI: 0.358239
Right branch Create node [21]. Number of n1: 45. Number of n2: 55. GINI: 0.495
Left branch Create node [22]. Number of n1: 9. Number of n2: 3. GINI: 0.375
Right branch Create node [23]. Number of n1: 36. Number of n2: 52. GINI: 0.483471
Right branch Create node [24]. Number of n1: 80. Number of n2: 186. GINI: 0.4206
Left branch Create node [25]. Number of n1: 14. Number of n2: 90. GINI: 0.232988
Left branch Create node [26]. Number of n1: 2. Number of n2: 4. GINI: 0.444444
Right branch Create node [27]. Number of n1: 12. Number of n2: 86. GINI: 0.21491
Right branch Create node [28]. Number of n1: 66. Number of n2: 96. GINI: 0.482853
Left branch Create node [29]. Number of n1: 45. Number of n2: 47. GINI: 0.499764
Left branch Create node [30]. Number of n1: 5. Number of n2: 1. GINI: 0.277778
Right branch Create node [31]. Number of n1: 40. Number of n2: 46. GINI: 0.497566
Right branch Create node [32]. Number of n1: 21. Number of n2: 49. GINI: 0.42
Accuracy: 0.720313
DOT file generated.

*/
