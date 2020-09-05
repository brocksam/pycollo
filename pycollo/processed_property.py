



def processed_property(name, **kwargs):

	storage_name = f"_{name}"
	expected_type = kwargs.get('type')
	optional = kwargs.get('optional', False)
	default = kwargs.get('default', False)
	iterable_allowed = kwargs.get('iterable_allowed', False)
	cast_to_type = kwargs.get('cast')
	len_sequence = kwargs.get('len')
	max_value = kwargs.get('max')
	min_value = kwargs.get('min')
	exclusive = kwargs.get('exclusive', False)
	private = kwargs.get('private', False)
	post_method = kwargs.get('method')

	@property
	def prop(self):
		return getattr(self, storage_name)

	@prop.setter
	def prop(self, value):

		if private:
			raise AttributeError

		if expected_type is not None:
			if iterable_allowed:
				if isinstance(value, Iterable):
					value = tuple([check_type(val) for val in value])
				else:
					value = (check_type(value), )
			else:
				value = check_type(value)

		if min_value is not None:
			value = check_min(value)

		if max_value is not None:
			value = check_max(value)

		if len_sequence is not None:
			value = check_len(value)

		if post_method is not None:
			value = apply_method(value)

		setattr(self, storage_name, value)


	def check_type(value):
		if isinstance(value, expected_type):
			return value
		elif optional and value is None:
			if default:
				return expected_type()
			else:
				return None
		elif cast_to_type:
			return cast_type(value)
		else:
			msg = (f"`{name}` must be a {repr(expected_type)}, instead got a "
				f"{repr(type(value))}.")
			raise TypeError(msg)

	def cast_type(value):
		cast_str = f"processed_value = {expected_type.__name__}({value})"
		try:
			exec(cast_str)
		except (ValueError, TypeError) as e:
			msg = (f"`{name}` must be a {repr(expected_type)}, instead got a "
				f"{repr(type(value))} which cannot be cast.")
			raise e(msg)
		return locals()['processed_value']

	def check_min(value):
		if exclusive:
			if value <= min_value:
				msg = (f"`{name}` must be greater than {min_value}.")
				raise ValueError(msg)
		else:
			if value < min_value:
				msg = (f"`{name}` must be greater than or equal to {min_value}.")
				raise ValueError(msg)
		return value

	def check_max(value):
		if exclusive:
			if value >= max_value:
				msg = (f"`{name}` must be less than {max_value}.")
				raise ValueError(msg)
		else:
			if value < max_value:
				msg = (f"`{name}` must be less than or equal to {max_value}.")
				raise ValueError(msg)
		return value

	def check_len(value):
		if len() != len_sequence:
			msg = (f"`{name}` must be a sequence of length {len_sequence}.")
			raise ValueError(msg)

	def apply_method(value):
		if optional and value is None:
			return None
		return post_method(value)

	return prop


