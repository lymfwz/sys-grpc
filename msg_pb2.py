# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: msg.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tmsg.proto\x12\x03msg\"\x1a\n\nMsgRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x1a\n\x0bMsgResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t2;\n\nMsgService\x12-\n\x06GetMsg\x12\x0f.msg.MsgRequest\x1a\x10.msg.MsgResponse\"\x00\x42)\n\x1b\x63om.example.spbclient.protoB\x08MsgProtoP\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'msg_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\033com.example.spbclient.protoB\010MsgProtoP\000'
  _globals['_MSGREQUEST']._serialized_start=18
  _globals['_MSGREQUEST']._serialized_end=44
  _globals['_MSGRESPONSE']._serialized_start=46
  _globals['_MSGRESPONSE']._serialized_end=72
  _globals['_MSGSERVICE']._serialized_start=74
  _globals['_MSGSERVICE']._serialized_end=133
# @@protoc_insertion_point(module_scope)
