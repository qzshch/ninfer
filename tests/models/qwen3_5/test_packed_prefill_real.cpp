#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/execution/text.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
namespace q = models::qwen3_5;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void append(std::vector<std::byte>& out, const Tensor& t) {
    auto n=out.size(); out.resize(n+t.bytes());
    CUDA_CHECK(cudaMemcpy(out.data()+n,t.data,t.bytes(),cudaMemcpyDeviceToHost));
}
std::vector<std::byte> snapshot(q::detail::ProgramImpl& p,int slot,int hidden_offset,int count) {
    std::vector<std::byte> out;
    auto& s=p.state_images->linear();
    for(std::uint32_t layer=0;layer<s.layer_count();++layer) {
        append(out,s.conv_slot(layer,slot)); append(out,s.recurrent_slot(layer,slot));
    }
    append(out,p.prefill_hidden.slice(1,hidden_offset,count));
    return out;
}
std::vector<float> values(const Tensor& tensor) {
    std::vector<std::byte> bytes;
    append(bytes,tensor);
    std::vector<float> out;
    if(tensor.dtype==DType::FP32) {
        out.resize(bytes.size()/sizeof(float));
        std::memcpy(out.data(),bytes.data(),bytes.size());
    } else {
        require(tensor.dtype==DType::BF16,"unexpected reference tensor dtype");
        out.resize(bytes.size()/sizeof(std::uint16_t));
        for(std::size_t i=0;i<out.size();++i) {
            std::uint16_t bf16;std::memcpy(&bf16,bytes.data()+2*i,sizeof(bf16));
            const std::uint32_t bits=static_cast<std::uint32_t>(bf16)<<16;
            std::memcpy(&out[i],&bits,sizeof(bits));
        }
    }
    return out;
}
std::vector<std::vector<float>> numeric_snapshot(q::detail::ProgramImpl& p,int slot,int offset,int count) {
    std::vector<std::vector<float>> out;
    auto& s=p.state_images->linear();
    for(std::uint32_t layer=0;layer<s.layer_count();++layer) {
        out.push_back(values(s.conv_slot(layer,slot)));
        out.push_back(values(s.recurrent_slot(layer,slot)));
    }
    out.push_back(values(p.prefill_hidden.slice(1,offset,count)));
    return out;
}
void compare_serial(const std::vector<std::vector<float>>& packed,
                    const std::vector<std::vector<float>>& serial) {
    require(packed.size()==serial.size(),"serial reference lost a tensor");
    double worst=0;bool within_gate=true;
    for(std::size_t tensor=0;tensor<serial.size();++tensor) {
        require(packed[tensor].size()==serial[tensor].size(),"serial reference shape mismatch");
        double error=0,scale=0,max_error=0;
        for(std::size_t i=0;i<serial[tensor].size();++i) {
            const double ref=serial[tensor][i],got=packed[tensor][i];
            require(std::isfinite(ref)&&std::isfinite(got),"nonfinite packed or serial state");
            const double d=got-ref;error+=d*d;scale+=ref*ref;max_error=std::max(max_error,std::abs(d));
        }
        const double relative=std::sqrt(error/std::max(scale,1e-30));
        if(scale>1e-20) {
            worst=std::max(worst,relative);
            within_gate=within_gate&&relative<=.025;
        } else within_gate=within_gate&&max_error<=1e-4;
        if ((scale>1e-20 && relative>.025) || (scale<=1e-20 && max_error>1e-4))
            std::cout<<"reference tensor "<<tensor<<" relative RMS "<<relative
                     <<" max abs "<<max_error<<'\n';
    }
    std::cout<<"serial represented-weight reference: worst tensor relative RMS "<<worst<<'\n';
    require(within_gate,"packed state/hidden drift exceeds serial numerical gate");
}
std::vector<std::byte> kv_snapshot(q::detail::ProgramImpl& p,int row,int pages) {
    auto& cache=p.decoder->text_kv;
    auto table=cache.execution_tables().matrix().slice(1,row,1);
    std::vector<std::int32_t> physical(pages);
    CUDA_CHECK(cudaMemcpy(physical.data(),table.data,pages*sizeof(std::int32_t),cudaMemcpyDeviceToHost));
    std::vector<std::byte> out;
    const auto& pool=cache.page_pool();
    for(std::size_t i=0;i<pool.plane_count();++i) {
        const auto& plane=pool.plane(i);
        for(auto page:physical) {
            require(page>=0,"test KV page not mapped");
            if(pool.geometry().device_plane_order==PagedKVPlaneOrder::PageMajor)
                append(out,plane.slice(3,page,1));
            else for(int head=0;head<plane.ne[3];++head)
                append(out,plane.slice(3,head,1).slice(2,page,1));
        }
    }
    return out;
}
void run(const char* artifact) {
    DeviceContext device;
    models::LoadOptions load;
    auto model=q::load_model(artifact,load,device);
    q::execution::Parameters parameters(*model);
    EngineOptions options;
    options.max_context=4096; options.prefill_chunk=4096;
    options.kv_capacity=KvCapacityPolicy::explicit_capacity(12288);
    options.max_concurrency=3; options.use_cuda_graph=false;
    options.context_cache.device_state_slots=0; options.context_cache.host_state_slots=0;
    options.context_cache.host_kv_capacity_bytes=0;
    options.context_cache.max_private_continuations=3;
    options.context_cache.max_shared_prefixes=0;
    options.context_cache.max_long_anchors_per_continuation=0;
    auto planner=q::make_sequence_planner(parameters,device,options);
    auto plan=std::move(planner).finalize(192);
    q::detail::ProgramImpl p(parameters,*plan.impl_,device,{});
    std::array<q::detail::KVAddressSpaceHandle,3> addresses;
    std::array<int,3> slots;
    for(int row=0;row<3;++row) {
        auto state=p.state_store->reserve_reset(device.stream); require(bool(state),"missing state slot");
        slots[row]=p.state_store->physical_slot(*state);
        auto address=p.text_kv_addresses->create_active(64,row,device.stream);
        require(bool(address),"missing KV row"); addresses[row]=*address;
        p.text_kv_addresses->ensure_mapped_to_tokens(*address,4096,device.stream);
    }
    std::array<std::vector<int>,3> tokens;
    for(int row=0;row<3;++row) {
        tokens[row].resize(4096);
        for(int i=0;i<4096;++i) tokens[row][i]=1000+row*200+(17*i)%113;
    }
    auto card=[&](int row){
        auto c=std::make_unique<q::execution::TextContext>(device,parameters,p.work,
            p.decoder->text_kv.execution_view(p.text_kv_addresses->execution_row(addresses[row])),
            p.state_images->linear(),p.io,p.prefill_hidden,4096,0,q::PagedKVCacheView{},&p.decoder->text_kv,nullptr);
        c->set_linear_state_slots(slots[row],slots[row]); return c;
    };
    auto first=card(0),second=card(1);
    std::array<q::execution::PackedPrefillSegment,2> segments{{
        {.context=first.get(),.prompt=tokens[0],.tokens=1024,.kv_table_row=p.text_kv_addresses->bound_row(addresses[0])},
        {.context=second.get(),.prompt=tokens[1],.tokens=1536,.kv_table_row=p.text_kv_addresses->bound_row(addresses[1])}}};
    auto a=first->prefill_packed(segments);
    require(a.size()==2&&a[0].processed_tokens==1024&&a[1].processed_tokens==1536,
            "ragged packed lengths lost");
    auto frozen=snapshot(p,slots[0],0,1024);
    const auto packed_numeric=numeric_snapshot(p,slots[0],0,1024);
    const auto row0=p.text_kv_addresses->bound_row(addresses[0]);
    auto frozen_kv=kv_snapshot(p,row0,16);
    auto other=snapshot(p,slots[1],1024,1536);
    p.state_images->linear().zero_all(device.stream);
    CUDA_CHECK(cudaMemcpyAsync(p.io.text_kv_table_row.data,&row0,sizeof(row0),
                               cudaMemcpyHostToDevice,device.stream));
    auto serial=first->prefill_chunk(tokens[0],0,1024,false);(void)serial;
    compare_serial(packed_numeric,numeric_snapshot(p,slots[0],0,1024));
    // Same shape and first row, entirely different second row: no state or activation coupling.
    for(auto& token:tokens[1]) token+=379;
    p.state_images->linear().zero_all(device.stream);
    auto b=first->prefill_packed(segments); (void)b;
    require(snapshot(p,slots[0],0,1024)==frozen,"another prompt changed first-row state/hidden");
    require(kv_snapshot(p,row0,16)==frozen_kv,"another prompt overwrote first-row KV pages");
    require(snapshot(p,slots[1],1024,1536)!=other,"changed second prompt did not change its own state");
    // Exercise the public reject path before any GPU work.
    segments[1].kv_table_row=segments[0].kv_table_row;
    bool rejected=false;
    try { auto invalid=first->prefill_packed(segments); (void)invalid; }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"aliased mutable KV row was accepted");
    segments[1].kv_table_row=p.text_kv_addresses->bound_row(addresses[1]);
    auto third=card(2);
    std::array<q::execution::PackedPrefillSegment,3> triple{{segments[0],segments[1],
        {.context=third.get(),.prompt=tokens[2],.tokens=1024,.kv_table_row=p.text_kv_addresses->bound_row(addresses[2])}}};
    p.state_images->linear().zero_all(device.stream);
    auto c=first->prefill_packed(triple); (void)c;
    auto first_frozen=snapshot(p,slots[0],0,1024);
    auto second_frozen=snapshot(p,slots[1],1024,1536);
    auto first_kv=kv_snapshot(p,row0,16);
    const auto row1=p.text_kv_addresses->bound_row(addresses[1]);
    auto second_kv=kv_snapshot(p,row1,24);
    for(auto& token:tokens[2]) token+=557;
    p.state_images->linear().zero_all(device.stream);
    auto d=first->prefill_packed(triple); (void)d;
    require(snapshot(p,slots[0],0,1024)==first_frozen&&snapshot(p,slots[1],1024,1536)==second_frozen,
            "third packed prompt changed another row's state/hidden");
    require(kv_snapshot(p,row0,16)==first_kv&&kv_snapshot(p,row1,24)==second_kv,
            "third packed prompt overwrote another row's KV pages");
    std::array<q::execution::PackedPrefillSegment,3> odd{{
        {.context=first.get(),.prompt=tokens[0],.tokens=127,.kv_table_row=row0},
        {.context=second.get(),.prompt=tokens[1],.tokens=191,.kv_table_row=row1},
        {.context=third.get(),.prompt=tokens[2],.tokens=95,
         .kv_table_row=p.text_kv_addresses->bound_row(addresses[2]),.rope_delta=37}}};
    p.state_images->linear().zero_all(device.stream);
    auto odd_a=first->prefill_packed(odd);(void)odd_a;
    auto odd_first=snapshot(p,slots[0],0,127);
    const auto odd_numeric=numeric_snapshot(p,slots[0],0,127);
    auto odd_second=snapshot(p,slots[1],127,191);
    auto odd_first_kv=kv_snapshot(p,row0,2);
    auto odd_second_kv=kv_snapshot(p,row1,3);
    p.state_images->linear().zero_all(device.stream);
    CUDA_CHECK(cudaMemcpyAsync(p.io.text_kv_table_row.data,&row0,sizeof(row0),
                               cudaMemcpyHostToDevice,device.stream));
    auto odd_serial=first->prefill_chunk(tokens[0],0,127,false);(void)odd_serial;
    compare_serial(odd_numeric,numeric_snapshot(p,slots[0],0,127));
    for(auto& token:tokens[2])token+=41;
    odd[2].rope_delta=53;
    p.state_images->linear().zero_all(device.stream);
    auto odd_b=first->prefill_packed(odd);(void)odd_b;
    require(snapshot(p,slots[0],0,127)==odd_first&&snapshot(p,slots[1],127,191)==odd_second,
            "odd tail or another row's RoPE delta leaked across lanes");
    require(kv_snapshot(p,row0,2)==odd_first_kv&&kv_snapshot(p,row1,3)==odd_second_kv,
            "odd tail overwrote another row's physical KV pages");
    for(auto address:addresses)
        require(p.text_kv_addresses->release_after_deactivate(address),"KV lease was not returned");
    std::cout<<"packed 1024/1536, 1024/1536/1024 and 127/191/95: prompt/state/KV invariance and alias rejection passed\n";
}
}
int main() {
    const char* artifact=std::getenv("NINFER_TEST_ARTIFACT");
    if(!artifact||!*artifact) return 77;
    try { run(artifact); return 0; }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
