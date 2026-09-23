#include<vector>
#include<stdexcept>
#include<cstddef>
#include<iostream>
#include<cassert>
#include<functional>
#include<memory>
#include<utility>
#include<unordered_set>


enum class Operation{
    leaf,
    add,
    subtract,
    multiply,
    divide,
    sum,
    matmul
};

struct TensorNode{
    //Dimensions of the Node
    std::vector<size_t> shape;
    std::vector<double> data;
    std::vector<double> grad;

    Operation operation = Operation::leaf;
    std::vector<std::shared_ptr<TensorNode>> parents;

    std::vector<size_t> left_strides;
    std::vector<size_t> right_strides;
};

class Tensor{
private:
    std::shared_ptr<TensorNode> m_node;
    void buildTopology(TensorNode* node, std::unordered_set<TensorNode*>& visited, std::vector<TensorNode*>& topology){
        if(!visited.insert(node).second){
            return;
        }

        for(const auto& parent : node->parents){
            buildTopology(parent.get(), visited, topology);
        }
        topology.push_back(node);
    }

    static void backwardElementwise(TensorNode* output, const std::function<std::pair<double, double>(double, double)>& localDerivatives){
        TensorNode* left = output->parents[0].get();
        TensorNode* right = output->parents[1].get();
        
        std::vector<size_t> coordinate(output->shape.size(), 0);

        for(size_t flat = 0, end = output->data.size(); flat < end; ++flat){
            const size_t leftIndex = strideOffset(coordinate, output->left_strides);
            const size_t rightIndex = strideOffset(coordinate, output->right_strides);

            const double gradient = output->grad[flat];

            const auto [leftDerivative, rightDerivative] = localDerivatives(left->data[leftIndex],right->data[rightIndex]);

            left->grad[leftIndex] += gradient * leftDerivative;
            right->grad[rightIndex] += gradient * rightDerivative;

            advanceCoordinate(coordinate, output->shape);
        }
    }


    size_t flatIndex(const std::vector<size_t>& idx) const{
        if(idx.size() != getRank()){
            throw std::invalid_argument("Numbrr of indices must match tensor rank");
        }
        
        for(size_t axis = 0, end = getRank(); axis < end; ++axis){
             if(idx[axis] >= m_node->shape[axis]){
                throw std::out_of_range{"Tensor index is outside its dimension"};
            }
        }
        
        size_t flatIndex = 0;
        size_t stride = 1;
         
         //iterate backwards through the tensor
        for(size_t axis = getRank(); axis > 0; --axis){
            const size_t currentAxis = axis - 1;
            
            //The index for the dimensions is 0 indexed so the index id should be one less than the index number 
            //handle overflow problem with forloop onlines 65-70
            flatIndex += idx[currentAxis] * stride;
            stride *= m_node->shape[currentAxis];
        }
        return flatIndex;
    }
    
    [[nodiscard]] Tensor elementwiseBinary(const Tensor& rhs, Operation kind, const std::function<double(const double, const double)>& operation) const {
        const std::vector<size_t> resultShape = broadcastShape(m_node->shape, rhs.m_node->shape);
        std::vector<size_t> leftStrides = effectiveStrides(m_node->shape, strides(), resultShape.size());
        std::vector<size_t> rightStrides = effectiveStrides(rhs.m_node->shape, rhs.strides(), resultShape.size());

        size_t resultNumEl =1;
        for(const size_t dim : resultShape){
            resultNumEl *= dim;
        }
        std::vector<double> resultData(resultNumEl);
        std::vector<size_t> coordinate(resultShape.size(), 0);

        for(size_t flat = 0; flat < resultNumEl; ++flat){
            const size_t leftIndex = strideOffset(coordinate, leftStrides);
            const size_t rightIndex = strideOffset(coordinate, rightStrides);

            resultData[flat] = operation(m_node->data[leftIndex], rhs.m_node->data[rightIndex]);
            advanceCoordinate(coordinate, resultShape);
        }
        Tensor result(std::move(resultShape), std::move(resultData));
        result.m_node->operation = kind;
        result.m_node->parents = {m_node, rhs.m_node};
        result.m_node->left_strides = std::move(leftStrides);
        result.m_node->right_strides = std::move(rightStrides);
        return result;
    }

    [[nodiscard]] static size_t strideOffset(const std::vector<size_t>& coordinate,const std::vector<size_t>& effectiveStrides){
        size_t index = 0;
        for(size_t axis = 0, end = coordinate.size(); axis < end; ++axis){
            index += coordinate[axis] * effectiveStrides[axis];
        }
        return index;
    }

    static void advanceCoordinate(std::vector<size_t>& coordinate, const std::vector<size_t>& shape){
        for(size_t axis = coordinate.size(); axis-- > 0;){
            if(++coordinate[axis] < shape[axis]){
                return;
            }
            coordinate[axis] = 0;   
        }
    }

    [[nodiscard]] static std::vector<size_t> broadcastShape(const std::vector<size_t>& left,const std::vector<size_t>& right){
        const size_t rank = std::max(left.size(), right.size());
        std::vector<size_t> result(rank);

        const size_t leftOffset = rank - left.size();
        const size_t rightOffset = rank - right.size();

        for(size_t axis = 0; axis < rank; ++axis){
            const bool leftHasAxis = axis >= leftOffset;
            const bool rightHasAxis = axis >= rightOffset;


            size_t leftDim = 1;
            if(leftHasAxis){
                const size_t leftAxis = axis - leftOffset;
                leftDim = left[leftAxis];
            }
            size_t rightDim = 1;
            if(rightHasAxis){
                const size_t rightAxis = axis - rightOffset;
                rightDim = right[rightAxis];
            }
            const bool sizesMatch = leftDim == rightDim;
            const bool leftCanStretch = leftDim == 1;
            const bool rightCanStretch = rightDim == 1;
            if(!sizesMatch && !leftCanStretch && !rightCanStretch){
                throw std::invalid_argument("Cannot broadcast - shape mismatch at dimension " + std::to_string(axis));
            }
            result[axis] = std::max(leftDim, rightDim);
        }
        return result;
    }

    [[nodiscard]] static std::vector<size_t> effectiveStrides(const std::vector<size_t>& shape,const std::vector<size_t>& ownStrides, const size_t targetRank){
        std::vector<size_t>result(targetRank, 0);
        const size_t offset = targetRank - shape.size();

        for(size_t axis = 0, end = shape.size(); axis < end; ++axis){
            if(shape[axis] != 1){
                result[offset + axis] = ownStrides[axis];
            }
        }
        return result;
    }
    
    [[nodiscard]] std::vector<size_t> strides() const {
        std::vector<size_t> results(m_node->shape.size());
        size_t stride = 1;
        
        //so instead of doingbthe hack used on lines 29 and 30
        //we are going to use the post-fix decrement on the condition
        //that way it will evaluate the index at the correct number 
        //but the index will be decremented by one when the condition matches
        //this prevents having to worry about underflowingbthe 
        //size_t. if we decrement it at zero and it wraps around it wont harm
        //the program because the condition won't pass because
        //because the index will be at zero therefore it won''t past the 
        //idx > 0 condition
        for(size_t axis = getRank(); axis-- > 0;){
            results[axis] = stride;
            stride *= m_node->shape[axis];
        }
        return results;
    }
    
public:
    Tensor( const std::vector<size_t>& shape, const std::vector<double>& data)
    	:m_node(std::make_shared<TensorNode>()){
            m_node->shape = std::move(shape);
            m_node->data = std::move(data);

    	    size_t expectedElements;
    	    bool hasZeroDim = false;
    	    
    	    for(const size_t& dimension : m_node->shape){
    	        //being overly strick with size_t max
    	        //took care if overflow
    	        if(dimension == 0){
    	            hasZeroDim = true;
    	            expectedElements = 0;
    	            break;
    	        }
    	    }
    	    
    	    if(!hasZeroDim){
    	       expectedElements = 1;
    	       for(const auto& dimension : m_node->shape ){
    	        	if(expectedElements > std::numeric_limits<size_t>::max() / dimension){
    	            throw std::overflow_error("Tensor count overflows size_t");
    	       	 }
    	        
    	        	expectedElements *= dimension;   
    	    	}
    	    }
    	   
    	    if(expectedElements != m_node->data.size()){
    	       throw std::invalid_argument("Tensor shape does not match its data");
    	    }
            m_node->grad.assign(getNumEle(), 0.0);
    }
    
    [[nodiscard]] const std::vector<size_t>& getShape() const noexcept{
        return m_node->shape;
    }
    
    [[nodiscard]] const std::vector<double>& getData() const noexcept{
        return m_node->data;
    }
    [[nodiscard]] const std::vector<double>& getGrad() const noexcept{
        return m_node->grad;
    }
    void backward(){
        if(getRank() != 0){
            throw std::invalid_argument("backward requires a scalar loss");
        }

        std::unordered_set<TensorNode*> visited;
        std::vector<TensorNode*> topology;
        buildTopology(m_node.get(), visited, topology);

        //now clear all gradients
        for(TensorNode* node : topology){
            std::fill(node->grad.begin(),node->grad.end(), 0.0);
        }

        m_node->grad[0] = 1.0;
        //now we'll work throgh the nodes backward using the operation tags to chose what todo
        for(auto it = topology.rbegin(), end = topology.rend(); it != end; ++it){
            TensorNode* output = *it;
            switch(output->operation){
                case Operation::leaf :
                    break;

                case Operation::add :
                    backwardElementwise(output, [](double, double) -> std::pair<double, double>{
            return std::pair{1.0, 1.0}; 
            });
                    break;

                case Operation::subtract :
                    backwardElementwise(output, [](double, double) -> std::pair<double, double>{
            return std::pair{1.0, -1.0}; 
            });
                    break;

                case Operation::multiply :
                    backwardElementwise(output, [](double left, double right) -> std::pair<double, double>{
            return std::pair{right, left}; 
            });
                    break;


                case Operation::divide :
                    backwardElementwise(output, [](double numerator, double denominator) -> std::pair<double, double>{
            return std::pair{1.0 / denominator, -numerator / (numerator * numerator)}; 
            });
                    break;


                default:
                    throw std::logic_error("backward rule not implemented yet");
            }
        }
    }
    [[nodiscard]] size_t getRank() const noexcept {
        return m_node->shape.size();
    }
    
    [[nodiscard]] size_t getNumEle() const noexcept {
        return m_node->data.size();
    }
    
    [[nodiscard]] size_t getDimension(const size_t axis){
        if(axis >= getRank()){
            throw std::out_of_range("Tensor axisis outside its rank");
        }
        return m_node->shape[axis];
    }
    double& at (const std::vector<size_t>& idx){
        return m_node->data[flatIndex(idx)];
    }
    
    [[nodiscard]] double at(const std::vector<size_t>& idx) const {
        return m_node->data[flatIndex(idx)];
     }
     [[nodiscard]] Tensor sum( ) const {
         double result = 0.0;
         for(const double value : m_node->data){
             result += value;
         }
         Tensor output({ }, { result});
         output.m_node->operation = Operation::sum;
         output.m_node->parents = {m_node};
         return output;
     }
     
     [[nodiscard]] Tensor matmul(const Tensor& other) const{
         if(getRank() != 2 || other.getRank() != 2){
             throw std::invalid_argument("Matmul requires 2 two-rank Tensors");
         }
         
         const size_t leftRow = m_node->shape[0];
         const size_t leftCol = m_node->shape[1];
         const size_t rightRow = other.m_node->shape[0];
         const size_t rightCol = other.m_node->shape[1];
         
         if(leftCol != rightRow){
             throw std::invalid_argument("Matmul inner dimensions must match");
         }
         
         const std::vector<size_t> resultShape{leftRow, rightCol};
         std::vector<double> resultData(leftRow * rightCol, 0.0);
         
         for(size_t row = 0; row < leftRow; ++row){
             for(size_t col = 0; col < rightCol; ++col){
                 //results[row][col] = current left row . dot . current right column
                 double sum = 0.0;
                 for(size_t idx = 0; idx < leftCol; ++idx){
                     sum += m_node->data[row * leftCol + idx] * other.m_node->data[idx * rightCol + col];
                 }
                 resultData[row * rightCol + col] = sum;
             }
         }
         
         Tensor result(
             std::move(resultShape),
             std::move(resultData)
         );
         result.m_node->operation = Operation::matmul;
         result.m_node->parents = {m_node, other.m_node};
         return result;
     }

     [[nodiscard]] Tensor mean() const {
         if(getNumEle() == 0){
             throw std::invalid_argument("Mean is undefined for an empty tensor");
         }
         return sum() / Tensor({ }, {static_cast<double>(getNumEle())});
     }
     
     [[nodiscard]] Tensor dot(const Tensor& other) const{
         if(getRank() != 1 || other.getRank() != 1){
             throw std::invalid_argument("Dot requires two rank-one Tensors");
         }
         
         if(m_node->shape != other.getShape()){
             throw std::invalid_argument("Dot requires two vectors of equal lengths");
         }
         //TODO(performance): benchmark a fused dot-productkernal that avoids
         //allocating and traversing an intermediate product tensor
         return (*this * other).sum();
     }
     
     [[nodiscard]] Tensor operator + (const Tensor& rhs) const {
        return elementwiseBinary(rhs, Operation::add, [] (const double lhs, const double rhs){
            return lhs + rhs; });
     }
     
     [[nodiscard]] Tensor operator - (const Tensor& rhs) const {
         return elementwiseBinary(rhs, Operation::subtract, [] (const double lhs, const double rhs){
            return lhs - rhs; });
     }
     
     [[nodiscard]] Tensor operator * (const Tensor& rhs) const {
        return elementwiseBinary(rhs, Operation::multiply, [] (const double lhs, const double rhs){
            return lhs * rhs; });
     }
     [[nodiscard]] Tensor operator / (const Tensor& rhs) const {
        return elementwiseBinary(rhs, Operation::divide, [] (const double lhs, const double rhs){
            if(rhs == 0.0){
                throw std::domain_error("division by zero");
            }
            return lhs / rhs; });
    }

};

[[nodiscard]] Tensor mseLoss(const Tensor& prediction, const Tensor& target){
    if(prediction.getShape() != target.getShape()){
        throw std::invalid_argument("mse loss requires equal shape");
    }
    Tensor residual = prediction - target;
    return (residual * residual).mean();
}


int main (){
      
      //shape  |  rank   |   elements  |  meaning
      //----------+-----------+------------------+-----------------------
      //   [  ]     |      0     |           1         |  scalar
      //----------+-----------+------------------+------------------------
      //  [ 1 ]    |      1     |           1         |  single vector
      //----------+-----------+------------------+-----------------------
      //  [ 0 ]   |      1      |           0         |   empty vector
      //----------+-----------+------------------+-----------------------
      // [5, 3]  |      2      |         15        |  5 x 3 matrix 
      //----------+-----------+------------------+-----------------------
      // [5, 0]  |      2      |          0         |  empty matrix 
      //----------+-----------+------------------+-----------------------
      
      std::vector<size_t> shape_1{2, 3};
    std::vector<double> data_1{0, 1, 2, 3, 4, 5};
    
    Tensor t(shape_1, data_1);
    
    std::puts("Tensor test 1");
    const auto& actual_shape_1 = t.getShape();
    assert(actual_shape_1 == shape_1);
	
	std::puts("Tensor test 2");
    const auto& actual_data_1 = t.getData();
    assert(actual_data_1 == data_1);
    
    std::puts("Tensor test 3");
    size_t actual_rank_1 = t.getRank();
    assert(actual_rank_1 == 2);
    
    std::puts("Tensor test 4");
    size_t actual_numel_1 = t.getNumEle();
    assert(actual_numel_1 == 6);
    
    std::puts("Tensor test 5");
    bool test_5_threw = false;
    try{
        Tensor({}, {});
    }catch(std::invalid_argument& error){
        assert(std::string(error.what()) == "Tensor shape does not match its data");
        test_5_threw = true;
    }
    assert(test_5_threw);
    Tensor scalar({ }, {5});
    assert(scalar.getRank() == 0);
    assert(scalar.getNumEle() == 1);
    assert(scalar.getShape() == std::vector<size_t>{ });
   
    std::puts("Tensor test 6");
    Tensor empty_matrix({1, 0}, { });
    assert(empty_matrix.getRank() == 2);
    assert(empty_matrix.getNumEle() == 0);
    assert(empty_matrix.getShape() == std::vector<size_t>({1, 0}));
   
   
   std::puts("Tensor test 7");
   bool test_7_threw = false;
   try{
        Tensor({2, 3}, {1, 2, 3});
    }catch(std::invalid_argument& error){
        assert(std::string(error.what()) == "Tensor shape does not match its data");
        test_7_threw = true;
    }
    assert(test_7_threw);
   
    std::puts("Tensor test 8");
    double val = t.at({1, 1});
    assert(val == 4);
   
    std::puts("Tensor test 9");
    bool test_9_threw = false;
    try{
        double val_9 = t.at({1, 2, 3});
    }catch(std::invalid_argument& error){
        assert(std::string(error.what()) == "Numbrr of indices must match tensor rank");
        test_9_threw = true;
    }
   assert(test_9_threw);
   
   std::puts("Tensor test 10");
    bool test_10_threw = false;
    try{
        double val_10 = t.at({2, 3});
    }catch(std::out_of_range& error){
        assert(std::string(error.what()) == "Tensor index is outside its dimension");
        test_10_threw = true;
    }
    assert(test_10_threw);
   
    std::puts("Tensor test 11");
    bool test_11_threw = false;
    try{
        Tensor overflow({std::numeric_limits<size_t>::max(), 2}, { });
    }catch(std::overflow_error& error){
        assert(std::string(error.what()) == "Tensor count overflows size_t");
        test_11_threw = true;
   }
   assert(test_11_threw);
   
   std::puts("Tensor test 12");
   Tensor zero_and_max({std::numeric_limits<size_t>::max(), 0}, {});
   assert(zero_and_max.getRank() == 2);
   assert(zero_and_max.getNumEle() == 0);
    
   std::puts("Tensor test 13");
   Tensor zero_and_max_3d({std::numeric_limits<size_t>::max(), 2, 0}, {});
   assert(zero_and_max_3d.getRank() == 3);
   assert(zero_and_max_3d.getNumEle() == 0);
    
    std::puts("Tensor test 14");
    Tensor dim_test_tensor({2, 0, 3}, {});
    assert(dim_test_tensor.getDimension(0) == 2);
    assert(dim_test_tensor.getDimension(1) == 0);
    assert(dim_test_tensor.getDimension(2) == 3);
    bool rejected_axis = false;
    try{
       auto d = dim_test_tensor.getDimension(3);
    }catch(const std::out_of_range& ){
       rejected_axis = true;
   }
   assert(rejected_axis);
   
   std::puts("Tensor test 15");
   Tensor mutation_tensor({2, 3}, {0,1,2,3,4,5});
   assert(mutation_tensor.at({0, 0}) == 0);
   mutation_tensor.at({0, 0}) = 5;
   assert( mutation_tensor.at({0, 0}) == 5);
   
   std::puts("Tensor test 16");
   Tensor scalar_16({ }, {10});
   assert(10 == scalar_16.at( { }));
   
   std::puts("Tensor test 17");
   assert(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}).sum( ).at({ })== 21.0);
   assert(Tensor({ }, {7.0}).sum( ).at({ }) == 7.0);
   assert(Tensor({1}, {7.0}).sum( ).at({ }) == 7.0);
   assert(Tensor({0}, { }).sum( ).at({ }) == 0);
   assert(Tensor({2, 0, 3}, { }).sum( ).at({ }) == 0);
   
   std::puts("Tensor test 18");
   const Tensor lhs({3}, {1.0, 2.0, 3.0});
   const Tensor rhs({3}, {10.0, 20.0, 30.0});
   
   const Tensor added = lhs + rhs;
   assert(added.getShape() == std::vector<size_t>({3}));
   assert(added.getData() == std::vector<double>({11.0, 22.0, 33.0}));
    assert(lhs.getData() == std::vector<double>({1.0, 2.0, 3.0}));
    assert(rhs.getData() == std::vector<double>({10.0, 20.0, 30.0}));
    
    const Tensor scalar_sum = Tensor({ }, {2.0}) + Tensor({ }, {3.0});
    assert(scalar_sum.getRank() == 0);
    assert(scalar_sum.at({ }) == 5.0);
    
    const Tensor empty_sum = Tensor({0}, { }) + Tensor({0}, { });
    assert(empty_sum.getShape() == std::vector<size_t>({0}));
    assert(empty_sum.getNumEle() == 0);
    
    std::puts("Tensor test 19");
    bool rejected_mismatched_shape = false;
    try{
        const auto invalid = Tensor({3}, {1.0, 2.0, 3.0}) + Tensor({1, 2}, {3.0, 4.0});
    }catch(const std::invalid_argument& ){
        rejected_mismatched_shape = true;
   }
   assert(rejected_mismatched_shape);
   
   std::puts("Tensor test 20");
   const Tensor arithematic_left({3}, {2.0, 3.0, 4.0});
   const Tensor arithematic_right({3}, {5.0, 6.0, 7.0});
    
    const Tensor arithematic_sum = arithematic_left + arithematic_right;
    const Tensor arithematic_dif = arithematic_left - arithematic_right;
    const Tensor arithematic_prod = arithematic_left * arithematic_right;
    assert(arithematic_sum.getData( ) == std::vector<double>({7.0, 9.0, 11.0}));
    assert(arithematic_dif.getData( ) == std::vector<double>({-3.0, -3.0, -3.0}));
    assert(arithematic_prod.getData( ) == std::vector<double>({10.0, 18.0, 28.0}));
    
    std::puts("Tensor test 21");
    const Tensor dot_results = Tensor({3}, {2.0, 3.0, 4.0}).dot(Tensor({3}, {5.0, 6.0, 7.0}));
    assert(dot_results.getRank() == 0);
    assert(dot_results.at({ }) == 56.0);
    
    const Tensor empty_dot = Tensor({0}, { }).dot(Tensor({0}, { }));
    assert(empty_dot.getRank() == 0);
    assert(empty_dot.at({ }) == 0);
    
    bool rejected_matrix_dot = false;
    try{
        const Tensor invalid = Tensor({1, 2}, {1.0, 2.0}).dot(Tensor({1, 2}, {3.0, 4.0}));
    }catch(const std::invalid_argument&){
        rejected_matrix_dot = true;
   }
   assert(rejected_matrix_dot);
   
   bool rejected_unequal_length = false;
    try{
        const Tensor invalid = Tensor({2}, {1.0, 2.0}).dot(Tensor({3}, {3.0, 4.0, 5.0}));
    }catch(const std::invalid_argument&){
         rejected_unequal_length= true;
   }
   assert(rejected_unequal_length);
    
    std::puts("Tensor test 22");
    //features = [4.0, 3.0, 2.0]
    //weights = [0.5, -1.0, 2.0]
    //scaled = [2.0, -3.0, 4.0]
    //weighted_sum = 3.0
    //bias = 0.5
    //prediction = 3.0 + 0.5 = 3.5
    //pred = weight.dot(features) + bias
    const Tensor features({3}, {4.0, 3.0, 2.0});  
    const Tensor weights1({3}, {0.5, -1.0, 2.0});
    const Tensor bias({ }, {0.5});
    
    const Tensor prediction = weights1.dot(features) + bias;
    
    assert(prediction.getRank() == 0);
    assert(prediction.at({ }) == 3.5);
    
    std::puts("Tensor test 23");
    const Tensor input({2, 3}, 
    						{
        						4.0, 3.0, 2.0,
        						1.0, 2.0, 0.5			
        											});
        											
    const Tensor weights2({3, 1}, 
    											{
    											    0.5,
    											    -1.0,
    											    2.0
    											          });
   const Tensor matmul23 = input.matmul(weights2);
   
   assert((matmul23.getShape() == std::vector<size_t>{2, 1}));
  
   assert((matmul23.getData() == std::vector<double>{3.0, -0.5}));

   const Tensor pred_23 = matmul23 + bias;
   assert((pred_23.getData() == std::vector<double>{3.5, 0.0}));

   const Tensor targets({2, 1}, {2.5, 1});
   const Tensor residuals = pred_23 - targets;
   assert((residuals.getData() == std::vector<double>{1.0, -1.0}));

   const Tensor residuals_total = residuals.sum();

   const Tensor squared_residuals = residuals * residuals;
assert((squared_residuals.getData() ==std::vector<double>{1.0, 1.0} ));

   const Tensor total_squared_error = squared_residuals.sum();
   assert((total_squared_error.sum().at({}) == 2));

   const Tensor mean_squared_error = squared_residuals.mean();
   assert(mean_squared_error.getRank() == 0);
   assert(mean_squared_error.sum().at({ }) == 1.0);


    std::puts("Tensor test 24");
    const Tensor loss = mseLoss(pred_23, targets);
    assert(loss.getRank() == 0);
    assert(loss.at({ }) == 1.0);

    const Tensor perfect_loss = mseLoss(pred_23, pred_23);
    assert(perfect_loss.at({ }) == 0.0);

    
   std::puts("Successful!"); 
}
