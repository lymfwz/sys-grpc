# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import msg_pb2 as msg__pb2


class MsgServiceStub(object):
    """@7 定义服务，用于描述要生成的API接口，类似于Java的业务逻辑接口类
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetMsg = channel.unary_unary(
                '/msg.MsgService/GetMsg',
                request_serializer=msg__pb2.MsgRequest.SerializeToString,
                response_deserializer=msg__pb2.MsgResponse.FromString,
                )


class MsgServiceServicer(object):
    """@7 定义服务，用于描述要生成的API接口，类似于Java的业务逻辑接口类
    """

    def GetMsg(self, request, context):
        """imgIdentify 方法名 ImgRequest 传入参数  ImgResponse 返回响应
        注意：这里是returns 不是return
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MsgServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetMsg': grpc.unary_unary_rpc_method_handler(
                    servicer.GetMsg,
                    request_deserializer=msg__pb2.MsgRequest.FromString,
                    response_serializer=msg__pb2.MsgResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'msg.MsgService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MsgService(object):
    """@7 定义服务，用于描述要生成的API接口，类似于Java的业务逻辑接口类
    """

    @staticmethod
    def GetMsg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/msg.MsgService/GetMsg',
            msg__pb2.MsgRequest.SerializeToString,
            msg__pb2.MsgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)