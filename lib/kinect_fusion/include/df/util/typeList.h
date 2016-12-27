#pragma once

namespace df {

template <int _I_>
struct IntToType {
    static constexpr int I = _I_;
};

struct NullType {

    static constexpr unsigned int Length = 0;

};

template <typename __Head__, typename __Tail__>
struct TypeList {

    typedef __Head__ Head;
    typedef __Tail__ Tail;

    static constexpr unsigned int Length = 1 + Tail::Length;

};

template <typename __TypeList__>
struct CompositedTypeListInstantiation {

    typename __TypeList__::Head head;
    CompositedTypeListInstantiation<typename __TypeList__::Tail> tail;

};

template <typename Head>
struct CompositedTypeListInstantiation<TypeList<Head,NullType> > {

    Head head;

};

} // namespace df

#define DF_TYPELIST1(type1) TypeList<type1,NullType>

#define DF_TYPELIST2(type1, type2) TypeList<type1,DF_TYPELIST1(type2)>

#define DF_TYPELIST3(type1, type2, type3) TypeList<type1,DF_TYPELIST2(type2,type3)>

#define DF_TYPELIST4(type1, type2, type3, type4) TypeList<type1,DF_TYPELIST3(type2,type3,type4)>

#define DF_TYPELIST5(type1, type2, type3, type4, type5) TypeList<type1,DF_TYPELIST4(type2,type3,type4,type5)>
