..
  class.rst

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members:
    :exclude-members: __weakref__, __repr__, __str__

    {% block methods %}
        {% if methods %}
            .. rubric:: {{ _('Attributes') }}

            .. autosummary::
                {% for item in attributes %}
                    ~{{ name }}.{{ item }}
                {%- endfor %}
        {% endif %}
        {% if methods %}
            .. rubric:: {{ _('Methods') }}

            .. autosummary::
                :nosignatures:
                {% for item in methods %}
                    ~{{ name }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}
